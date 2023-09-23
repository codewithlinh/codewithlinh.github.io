import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataloader, optimizer=None, learning_rate=0.0005, loss_function=None, device='cuda'):
        self.model = model.to(device)
        self.dataloader = dataloader

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        self.loss_function = loss_function
        self.device = device
        self.current_epoch = 0

    def train(self, epochs, print_every=1, writer=None):
        epoch_losses = []
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            if epoch % print_every == 0:
                print(f"======== Epoch: {epoch} ========")

            batch_losses = []
            for idx, data_dict in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                x1, x2 = data_dict['x1'], data_dict['x2']
                x1, x2 = x1.to(self.device), x2.to(self.device)

                batch_loss = self.train_iter(x1, x2)
                batch_losses.append(batch_loss.item())

            epoch_losses.append(np.mean(batch_losses))
            self.current_epoch += 1

            if epoch % print_every == 0:
                print(f'Average train loss: {np.mean(epoch_losses)}')

        return epoch_losses

    def train_iter(self, x1, x2, verbose=0):
        self.optimizer.zero_grad()
        embedding1, embedding2 = self.model(x1), self.model(x2)
        loss = self.loss_function(embedding1, embedding2)
        loss.backward()
        self.optimizer.step()

        return loss
