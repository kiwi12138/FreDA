"""
Builds upon: https://github.com/mr-eggplant/EATA
Corresponding paper: https://arxiv.org/abs/2204.02610
"""

import math
import logging

import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from methods.base import TTAMethod
from datasets.data_loading import get_source_loader

logger = logging.getLogger(__name__)
from methods.fourier_clustering import kmeans_torch, process_batch
import copy
from methods.other_utils import synthesize_data,initialize_centroids


def clone_model_with_normalizer(original_model):

    if isinstance(original_model, nn.Sequential):
        cloned_model = torch.nn.Sequential()
        for module in original_model:
            cloned_model.add_module(module.__class__.__name__, copy.deepcopy(module))

    else:
        cloned_model = type(original_model)()
        cloned_model.load_state_dict(copy.deepcopy(original_model.state_dict()),strict=False)

    return cloned_model.cuda()


class FreDA(TTAMethod):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.num_samples_update_1 = 0  # number of samples after First filtering, exclude unreliable samples
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = math.log(self.num_classes) * 0.40  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = cfg.EATA.D_MARGIN  # hyperparameter \epsilon for cosine similarity thresholding (Eqn. 5)

        self.current_model_probs = None  # the moving average of probability vector (Eqn. 4)
        self.fisher_alpha = cfg.EATA.FISHER_ALPHA  # trade-off \beta for two losses (Eqn. 8)
        self.fishers = None


        self.memory_pool_features = torch.empty((0, cfg.EATA.FFT_DIM), device='cuda')


        self.n_clusters = cfg.EATA.CLUSTER_NUM
        self.count = 0
        self.max_samples_per_cluster = cfg.EATA.CLUSTER_SIZE  # todo
        self.kmeans_size = cfg.EATA.KMEANS_SIZE
        self.cluster_sample_dict = {i: [] for i in range(self.n_clusters)}
        self.predict_kmeans_cluster = []
        self.true_corruption_labels = []

        self.cluster_models = [clone_model_with_normalizer(model) for _ in range(self.n_clusters)]
        for jj in range(self.n_clusters):
            for name, param in self.cluster_models[jj].named_parameters():
                param.requires_grad = True  # todo


        self.cluster_optimizer = [
            torch.optim.SGD(filter(lambda p: p.requires_grad, self.cluster_models[ii].parameters()), lr=cfg.EATA.LOCAL_LR) for ii in
            range(self.n_clusters)]

        self.global_model = clone_model_with_normalizer(model)
        if torch.cuda.device_count() > 1:
            print("Using Multiple GPU")
            self.model = nn.DataParallel(self.model)
            self.global_model = nn.DataParallel(self.global_model)
            self.cluster_models = [nn.DataParallel(cluster_model) for cluster_model in self.cluster_models]
        self.fedcount = 1
        self.comm_interval = cfg.EATA.COMM_INTERVAL

    def update_memory_pool(self, processed_features):
        self.memory_pool_features = torch.cat((self.memory_pool_features, processed_features), dim=0)
        selected_features = self.memory_pool_features[-self.kmeans_size:, :]
        predict_cluster_labels, updated_centroids = kmeans_torch(selected_features, self.centroids, n_iters=10)
        self.memory_pool_features = self.memory_pool_features[-self.kmeans_size:, :]
        return updated_centroids, predict_cluster_labels[-processed_features.shape[0]:]

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        Return: model outputs
        """
        imgs_test = x[0].cuda()
        original_indices = torch.arange(imgs_test.size(0))
        processed_features = process_batch(imgs_test)

        if self.count == 0:
            initial_centroids = initialize_centroids(processed_features,self.n_clusters)
            self.centroids = initial_centroids

        updated_centroids, cluster_assignments = self.update_memory_pool(processed_features)
        self.centroids = updated_centroids
        self.count += 1

        for img, cluster_idx, idx in zip(imgs_test, cluster_assignments, original_indices):
            cluster_idx = int(cluster_idx)
            self.cluster_sample_dict[cluster_idx].append((img.detach(), idx))
            if len(self.cluster_sample_dict[cluster_idx]) > self.max_samples_per_cluster:
                self.cluster_sample_dict[cluster_idx].pop(0)


        if self.fedcount % self.comm_interval == 0:
            global_params_sum = [torch.zeros_like(param) for param in self.model.parameters()]


        if any(len(samples) == self.max_samples_per_cluster for samples in self.cluster_sample_dict.values()):
            all_preds = []
            all_indices = []
            for cluster_idx in range(self.n_clusters):
                if len(self.cluster_sample_dict[cluster_idx]):
                    cluster_samples, indices = zip(*self.cluster_sample_dict[cluster_idx])
                    local_model = self.cluster_models[cluster_idx]

                    cluster_optimizer = self.cluster_optimizer[cluster_idx]
                    local_model.train()
                    cluster_samples_tensor = (torch.stack(cluster_samples))
                    local_preds = local_model(cluster_samples_tensor)

                    entropys = softmax_entropy(local_preds)


                    filter_ids_1 = torch.where(entropys < self.e_margin)
                    ids1 = filter_ids_1
                    ids2 = torch.where(ids1[0] > -0.1)
                    entropys = entropys[filter_ids_1] # filter redundant samples

                    if self.current_model_probs is not None:
                        cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0),
                                                                  local_preds[filter_ids_1].softmax(1), dim=1)
                        filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
                        entropys = entropys[filter_ids_2]
                        ids2 = filter_ids_2
                        updated_probs = update_model_probs(self.current_model_probs,
                                                           local_preds[filter_ids_1][filter_ids_2].softmax(1))
                    else:
                        updated_probs = update_model_probs(self.current_model_probs, local_preds[filter_ids_1].softmax(1))

                    if self.current_model_probs is None:
                        filter_ids_2 = torch.arange(len(ids1[0]))

                    if len(cluster_samples_tensor[filter_ids_1][filter_ids_2]) != 0:
                        synthetic_samples, random_indices = synthesize_data(cluster_samples_tensor, filter_ids_1,
                                                                            filter_ids_2, entropys, local_preds)
                        synthetic_preds = local_model(synthetic_samples)
                        selected_filtered_idx = local_preds[filter_ids_1][filter_ids_2][random_indices]
                        synthetic_loss = F.cross_entropy(synthetic_preds, selected_filtered_idx.argmax(1))
                    else:
                        synthetic_loss = 0.0

                    loss = entropys.mean(0) + 0.5 * synthetic_loss  # todo

                    if cluster_samples_tensor[ids1][ids2].size(0) != 0:
                        loss.backward()
                        cluster_optimizer.step()
                    else:
                        local_preds = local_preds.detach()

                    cluster_optimizer.zero_grad()


                    batch_numbers = int(torch.sum(cluster_assignments == cluster_idx))
                    if batch_numbers != 0:
                        with torch.no_grad():  # todo
                            local_preds = local_model(cluster_samples_tensor)
                        temp_preds_batch = local_preds[-batch_numbers:]
                        all_preds.append(temp_preds_batch)
                        all_indices.extend(indices[-batch_numbers:])


                    self.num_samples_update_1 += filter_ids_1[0].size(0)
                    self.num_samples_update_2 += entropys.size(0)
                    self.reset_model_probs(updated_probs)

                    total_samples = sum(len(samples) for samples in self.cluster_sample_dict.values())
                    if self.fedcount % self.comm_interval == 0:
                        cluster_sample_ratio = len(cluster_samples) / total_samples
                        with torch.no_grad():
                            for global_param, local_param in zip(global_params_sum, local_model.parameters()):
                                global_param += local_param.data * cluster_sample_ratio

            if self.fedcount % self.comm_interval == 0:
                with torch.no_grad():
                    for global_param, model_param in zip(global_params_sum, self.model.parameters()):
                        model_param.data = global_param.clone()

                    for local_model in self.cluster_models:
                        for local_param, global_param in zip(local_model.parameters(), self.model.parameters()):
                            local_param.data = global_param.clone()



            self.fedcount += 1
            torch.cuda.empty_cache()

            preds = torch.cat(all_preds, dim=0)
            sorted_preds = preds[torch.argsort(torch.tensor(all_indices))]

            return sorted_preds


        else:
            outputs = self.model(imgs_test)
            entropys = softmax_entropy(outputs)

            # filter unreliable samples
            filter_ids_1 = torch.where(entropys < self.e_margin)
            ids1 = filter_ids_1
            ids2 = torch.where(ids1[0] > -0.1)
            entropys = entropys[filter_ids_1]
            # filter redundant samples
            if self.current_model_probs is not None:
                cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0),
                                                          outputs[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
                entropys = entropys[filter_ids_2]
                ids2 = filter_ids_2
                updated_probs = update_model_probs(self.current_model_probs,
                                                   outputs[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(self.current_model_probs, outputs[filter_ids_1].softmax(1))

            if self.current_model_probs is None:
                filter_ids_2 = torch.arange(len(ids1[0]))

            if len(imgs_test[filter_ids_1][filter_ids_2]) != 0:
                synthetic_samples, random_indices = synthesize_data(imgs_test, filter_ids_1,
                                                                    filter_ids_2, entropys, outputs)
                synthetic_preds = self.model(synthetic_samples)
                selected_filtered_idx = outputs[filter_ids_1][filter_ids_2][random_indices]
                synthetic_loss = F.cross_entropy(synthetic_preds, selected_filtered_idx.argmax(1))
            else:
                synthetic_loss = 0.0
            loss = entropys.mean(0) + 0.5 * synthetic_loss

        # """
        # # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # # if x[ids1][ids2].size(0) != 0:
        # #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        # """
            if self.fishers is not None:
                ewc_loss = 0
                for name, param in self.model.named_parameters():
                    if name in self.fishers:
                        ewc_loss += self.fisher_alpha * (
                                    self.fishers[name][0] * (param - self.fishers[name][1]) ** 2).sum()
                loss += ewc_loss
            if imgs_test[ids1][ids2].size(0) != 0:
                loss.backward()
                self.optimizer.step()
            else:
                outputs = outputs.detach()
            self.optimizer.zero_grad()

            self.num_samples_update_1 += filter_ids_1[0].size(0)
            self.num_samples_update_2 += entropys.size(0)
            self.reset_model_probs(updated_probs)
            torch.cuda.empty_cache()

            return outputs

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_model_probs(self, probs):
        self.current_model_probs = probs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with eata."""
        # train mode, because eata optimizes the model to minimize entropy
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what eata updates
        self.model.requires_grad_(False)
        # configure norm for eata updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()  # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x / temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
