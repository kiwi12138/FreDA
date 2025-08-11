
import os
import json
import torch
import logging
from glob import glob
from typing import Optional, Sequence
import numpy as np
from robustbench.data import CORRUPTIONS, PREPROCESSINGS, load_cifar10c, load_cifar100c
from robustbench.loaders import CustomImageFolder, CustomCifarDataset
import random
# from classification.conf import cfg
logger = logging.getLogger(__name__)


def create_cifarc_dataset(
    dataset_name: str = 'cifar10_c',
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    domain = []
    x_test = torch.tensor([])
    y_test = torch.tensor([])
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]

    for cor in corruptions_seq:
        if dataset_name == 'cifar10_c':
            x_tmp, y_tmp = load_cifar10c(severity=severity,
                                         data_dir=data_dir,
                                         corruptions=[cor])
        elif dataset_name == 'cifar100_c':
            x_tmp, y_tmp = load_cifar100c(severity=severity,
                                          data_dir=data_dir,
                                          corruptions=[cor])
        else:
            raise ValueError(f"Dataset {dataset_name} is not suported!")

        x_test = torch.cat([x_test, x_tmp], dim=0)
        y_test = torch.cat([y_test, y_tmp], dim=0)
        domain += [cor] * x_tmp.shape[0]


    x_test = x_test.numpy().transpose((0, 2, 3, 1))
    y_test = y_test.numpy()
    if 'long_tail' in setting:
        if dataset_name == 'cifar10_c':
            class_num = 10
        elif dataset_name == 'cifar100_c':
            class_num = 100
        prob_per_class = []
        for cls_idx in range(class_num):
            prob_per_class.append(0.1 ** (cls_idx / (class_num - 1.0)))
        prob_per_class = np.array(prob_per_class) / sum(prob_per_class)
        img_per_class = prob_per_class * len(y_test)
        idx = []

        for c, num in enumerate(img_per_class):
            all_of_c = np.where(y_test == c)[0]
            idx.append(np.random.choice(all_of_c, int(num) + 1))
        idx = np.concatenate(idx)
        random.shuffle(idx)
        samples = [[x_test[i], y_test[i], domain[i]] for i in idx]

    else:
        samples = [[x_test[i], y_test[i], domain[i]] for i in range(x_test.shape[0])]

    return CustomCifarDataset(samples=samples, transform=transform)


def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
            class_to_idx = json.load(f)

        if n_examples != -1 or "correlated" in setting:
            # load test list containing all 50k image ids
            filename = os.path.join("datasets", "imagenet_list", "imagenet_val_ids_50k.txt")
        else:
            # load default test list from robustbench
            filename = os.path.join("robustbench", "data", "imagenet_test_image_ids.txt")

        # load file containing file ids
        with open(filename, 'r') as f:
            fnames = f.readlines()

        files = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            files += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = files

    return dataset_test
