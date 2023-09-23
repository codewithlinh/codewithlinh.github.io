import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from torch import nn
import torchsummary
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models

from dataloader import LabelContrastiveDataset
from losses import ContrastiveLoss
from trainers import Trainer


class TripletLoss(nn.Module):
    def __init__(self, margin, norm, miner, *args, **kwargs):
        super(TripletLoss, self).__init__(*args, **kwargs)
        self.loss = nn.TripletMarginLoss(margin=margin, p=norm)
        self.miner = miner

    def forward(self, x, y):
        a, p, n = self.miner(x, y)
        return self.loss(a, p, n)


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# %% Setup hyperparameters
N = 3000
EMBEDDING_SIZE = 2
DEVICE = 'cpu'
LEARNING_RATE, EPOCHS, MARGIN = 0.0005, 10, 1.0
NAME = 'MNIST_TRIPLET_LOSS_' + '_'.join([str(N), str(EMBEDDING_SIZE), str(LEARNING_RATE), str(EPOCHS), str(MARGIN)])

mean, std = 0.1307, 0.3081
transforms = transforms.Compose([transforms.Normalize((mean,), (std,))])
lcd = LabelContrastiveDataset(dataset_name='mnist', transform=None)
train_sampler = SubsetRandomSampler(range(int(N*0.9)))
test_sampler = SubsetRandomSampler(range(int(N*0.9), N))
siamese_train_loader = torch.utils.data.DataLoader(lcd, batch_size=None, sampler=train_sampler)

embedding_net = models.resnet18()
embedding_net.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3))
embedding_net.fc = nn.Linear(512, EMBEDDING_SIZE)
embedding_net.train()

triplet_loss = ContrastiveLoss(margin=1.0, mode='triplet')

trainer = Trainer(model=embedding_net, dataloader=siamese_train_loader, optimizer=None, learning_rate=LEARNING_RATE,
                  loss_function=triplet_loss, device=DEVICE)
losses = trainer.train(EPOCHS, print_every=1)
