import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEmbedding(nn.Module):
    def __init__(self, D_in, embedding=32):
        super(CNNEmbedding, self).__init__()
        self.D_in = D_in
        self.embedding = embedding
        self.conv1 = nn.Conv1d(1, 5, 3, 1)
        self.flat_size = (D_in // 2 - 1) * 5
        self.fc1 = nn.Linear(self.flat_size, embedding)

    def forward(self, x):
        x = x.view(-1, 1, self.D_in)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)

        x = x.view(-1, self.flat_size)
        x = self.fc1(x)

        return x


class FCNEmbedding(nn.Module):
    def __init__(self, D_in, H=256, embedding=32):
        super(FCNEmbedding, self).__init__()
        self.D_in = D_in
        self.embedding = embedding
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, embedding)

    def forward(self, x):
        x = x.view(-1, self.D_in)
        h_activate = F.relu(self.linear1(x))
        embedding = self.linear2(h_activate)
        return embedding


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        x1, x2, x3 = x
        return self.embedding_net(x1), \
               self.embedding_net(x2), \
               self.embedding_net(x3)


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, x,):
        anchor, positive, negative = x
        losses = self.dist(anchor, positive) - self.dist(anchor, negative)
        return F.relu(losses + 1.0)


class PairwiseNet(nn.Module):
    def __init__(self, embedding_net):
        super(PairwiseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        x1, x2 = x
        return self.embedding_net(x1), \
               self.embedding_net(x2)


class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()
        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, x,):
        a, b, d = x
        losses = self.dist(a, b) - d
        return losses.pow(2).mean()