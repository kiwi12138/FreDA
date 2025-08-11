import torch
from PIL import Image
import numpy as np
import os


def initialize_centroids(X, k, perturbation_scale=0.1):
    n_samples, n_features = X.shape

    if n_samples >= k:
        for i in range(1, k):
            centroids = torch.zeros((k, n_features)).cuda()
            # Choose the first centroid randomly from the data points
            centroids[0] = X[torch.randint(0, n_samples, (1,))]
            # Compute the distance from each point to the nearest centroid
            distances = torch.min(torch.stack([torch.norm(X - c, dim=1) for c in centroids[:i]]), dim=0).values

            # Compute the probability of each point being chosen as the next centroid
            probabilities = distances / torch.sum(distances)

            # Choose the next centroid randomly according to the computed probabilities
            next_centroid_index = torch.multinomial(probabilities, 1)
            centroids[i] = X[next_centroid_index]
    else:
        # centroids = torch.zeros((k, n_features)).cuda()
        indices = torch.randint(0, n_samples, (k,))
        centroids = X[indices].cuda()
        perturbation = perturbation_scale * torch.randn((k, n_features)).cuda()
        centroids += perturbation

    return centroids.cuda()


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def synthesize_data(cluster_samples_tensor,filter_ids_1,filter_ids_2,entropys,local_preds):

    filtered_samples_fft = torch.fft.fft2(cluster_samples_tensor[filter_ids_1][filter_ids_2])
    filtered_phases = torch.angle(filtered_samples_fft)
    filtered_amplitudes = torch.abs(filtered_samples_fft)


    random_indices = torch.randint(0, filtered_samples_fft.size(0), ((filtered_amplitudes.size(0)),))
    selected_amplitudes = filtered_amplitudes[random_indices]
    selected_phases = filtered_phases[random_indices]


    alpha = 0.1
    combine_coeffs = alpha * torch.randn_like(selected_amplitudes).cuda()
    combined_amplitude = (combine_coeffs + 1) * selected_amplitudes

    synthetic_fft = combined_amplitude * torch.exp(1j * selected_phases)
    synthetic_samples = torch.fft.ifft2(synthetic_fft).real

    synthetic_samples = torch.clamp(synthetic_samples, 0, 1)

    return synthetic_samples, random_indices.cuda()

