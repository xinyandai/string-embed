import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerCNN(nn.Module):
    def __init__(self, C, M, embedding=32):
        super(TwoLayerCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.conv1 = nn.Conv1d(1, 8, 3, 1, padding=1, bias=False)
        self.flat_size = M // 2  * C * 8
        self.fc1 = nn.Linear(self.flat_size, embedding)

    def forward(self, x):
        N, C, M  = x.shape
        x = x.view(N * C, 1, M)

        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)


        x = x.view(N, self.flat_size)
        x = self.fc1(x)

        return x


class FCNEmbedding(nn.Module):
    def __init__(self, C, M, H=256, embedding=32):
        super(FCNEmbedding, self).__init__()
        D_in = C * M
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
        losses = self.dist(anchor, positive) \
                 - self.dist(anchor, negative)
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