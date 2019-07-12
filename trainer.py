import math
import torch
import numpy as np
import torch.optim as optim

from networks import TripletNet
from networks import TripletLoss
from networks import CNNEmbedding as EmbeddingNet

def train_epoch(epoch, train_loader, device):
    N = len(train_loader)
    D = train_loader.D

    model = TripletNet(EmbeddingNet(D).to(device)).to(device)
    losser = TripletLoss()

    epochs = [0, 5]
    lrs = [0.01, 0.001]
    momentum = 0.9
    model.train()
    for epoch in range(epoch):
        for i_epoch, i_lr in zip(epochs, lrs):
            if epoch == i_epoch:
                optimizer = optim.SGD(model.parameters(), lr=i_lr,
                                      momentum=momentum, weight_decay=5e-4)
        agg = 0.0
        for idx in range(N):
            x = train_loader[idx]
            x = (i.to(device) for i in x)
            optimizer.zero_grad()
            output = model(x)
            loss = losser(output)
            loss.backward()
            optimizer.step()
            agg += loss.item()
            # print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, agg / float(N)))

    return model