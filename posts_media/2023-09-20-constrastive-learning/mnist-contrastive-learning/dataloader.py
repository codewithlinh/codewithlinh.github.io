import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as torch_datasets

import numpy as np
import random
import torch


class LabelContrastiveDataset(Dataset):
    def __init__(self, dataset_name='mnist', transform=None):
        super(LabelContrastiveDataset, self).__init__()

        if dataset_name == 'mnist':
            self.dataset = torch_datasets.MNIST(root='', train=True, transform=transform, download=True)

        self.labels_to_imgs = {}  # dict with labels as keys and file paths as values
        self.transform = transform
        for label in self.dataset.class_to_idx.values():
            self.labels_to_imgs[label] = [data[0] for data in self.dataset if data[1] == label]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        label_tensor = []

        # grab images with the same class as the anchor
        similar_imgs = len(self.labels_to_imgs[label])
        if similar_imgs > 2:
            similar_imgs_idx = random.choice(range(similar_imgs))  # random select one sample from same label
            selected_img = np.array(self.labels_to_imgs[label][similar_imgs_idx])
        else:
            raise NotImplementedError

        # transform original and random one
        if self.transform is not None:
            img = self.transform(img)
            selected_img = self.transform(selected_img)

        out_tensor_x1 = img[np.newaxis, ...]
        out_tensor_x2 = selected_img[np.newaxis, ...]
        label_tensor.append(int(label))

        # to form a batch, grab all the other classes
        all_labels = set(self.labels_to_imgs.keys())
        all_labels.discard(label)
        for negative in all_labels:  # take all other classes except anchor label
            dissimilar_imgs = len(self.labels_to_imgs[negative])

            # select 2 dissimilar images from other class as negative samples
            dissimilar_imgs_idx1 = random.choice(range(dissimilar_imgs))
            selected_dissimilar_imgs1 = np.array(self.labels_to_imgs[negative][dissimilar_imgs_idx1])
            dissimilar_imgs_idx2 = random.choice(range(dissimilar_imgs))
            while dissimilar_imgs_idx1 == dissimilar_imgs_idx2:
                dissimilar_imgs_idx2 = random.choice(range(dissimilar_imgs))
            selected_dissimilar_imgs2 = np.array(self.labels_to_imgs[negative][dissimilar_imgs_idx2])

            if self.transform is not None:
                selected_dissimilar_imgs1 = self.transform(selected_dissimilar_imgs1)
                selected_dissimilar_imgs2 = self.transform(selected_dissimilar_imgs2)

            selected_dissimilar_imgs1 = selected_dissimilar_imgs1[np.newaxis, ...]
            selected_dissimilar_imgs2 = selected_dissimilar_imgs2[np.newaxis, ...]
            out_tensor_x1 = torch.cat([torch.Tensor(out_tensor_x1), torch.Tensor(selected_dissimilar_imgs1)])
            out_tensor_x2 = torch.cat([torch.Tensor(out_tensor_x2), torch.Tensor(selected_dissimilar_imgs2)])

            label_tensor.append(int(negative))

        out_dict = {'x1': out_tensor_x1, 'x2': out_tensor_x2, 'labels': torch.Tensor(label_tensor)}
        return out_dict

