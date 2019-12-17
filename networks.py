import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerPool(nn.Module):
    def __init__(self, C, M, embedding=32):
        super(TwoLayerPool, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.pool = nn.Sequential(nn.MaxPool1d(16), nn.MaxPool1d(16),)

        self.flat_size = C * M // 256

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            N, C, M = x.shape
        else:
            N = 1
            C, M = x.shape

        x = x.view(N, C, M)
        x = self.pool(x)
        x = x.view(N, self.flat_size)

        return x


class MultiLayerCNN(nn.Module):
    def __init__(self, C, M, embedding=32):
        super(MultiLayerCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(8, 8, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
        )

        # Size after pooling
        self.flat_size = M // 1024 * C * 8
        print("self.flat_size ", self.flat_size)
        self.fc1 = nn.Linear(self.flat_size, embedding)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            N, C, M = x.shape
        else:
            N = 1
            C, M = x.shape
        x = x.view(N * C, 1, M)
        x = self.conv(x)
        # Size: (n * c, 8, M // 1024)
        # print(x.size())
        x = x.view(N, self.flat_size)
        x = self.fc1(x)

        return x


class TwoLayerCNN(nn.Module):
    def __init__(self, C, M, embedding=32):
        super(TwoLayerCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.conv1 = nn.Conv1d(1, 8, 3, 1, padding=1, bias=False)
        self.flat_size = M // 2 * C * 8
        self.fc1 = nn.Linear(self.flat_size, embedding)

    def forward(self, x):
        if len(x.shape) == 3:
            N, C, M = x.shape
        else:
            N = 1
            C, M = x.shape
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
        return self.embedding_net(x1), self.embedding_net(x2), self.embedding_net(x3)


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.dist = nn.PairwiseDistance(p=2)
        self.l, self.r = 1, 1
        self.Ls = {
            0: (0, 1),
            20: (0.2, 0.8),
            25: (0.5, 0.5),
            30: (0.8, 0.2),
            35: (1.0, 0),
        }

    def forward(self, x, dists, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        anchor, positive, negative = x
        pos_dist, neg_dist, pos_neg_dist = dists
        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        threshold = 0.1 + neg_dist - pos_dist
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)
        mse_loss = (
            (pos_embed_dist - pos_dist) ** 2
            + (neg_embed_dist - neg_dist) ** 2
            + (pos_neg_embed_dist - pos_neg_dist) ** 2
        )
        return self.l * rank_loss + self.r * mse_loss
