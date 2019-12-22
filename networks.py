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
    def __init__(self, C, M, embedding, channel):
        super(MultiLayerCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding

        self.conv = nn.Sequential(
            nn.Conv1d(1, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
        )

        # Size after pooling
        self.flat_size = M // 1024 * C * channel
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
        # print(x.size())
        x = x.view(N, self.flat_size)
        x = self.fc1(x)

        return x


class QuerylogCNN(nn.Module):
    def __init__(self, C, M, embedding, channel):
        super(QuerylogCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.conv = nn.Sequential(
            nn.Conv1d(1, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
            nn.Conv1d(channel, channel, 3, 1, padding=1, bias=False),
            nn.AvgPool1d(2),
        )

        # Size after pooling
        self.flat_size = M // 32 * C * channel
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
        # print(x.size())
        x = x.view(N, self.flat_size)
        x = self.fc1(x)

        return x


class TwoLayerCNN(nn.Module):
    def __init__(self, C, M, embedding, channel):
        super(TwoLayerCNN, self).__init__()
        self.C = C
        self.M = M
        self.embedding = embedding
        self.conv1 = nn.Conv1d(1, channel, 3, 1, padding=1, bias=False)
        self.flat_size = M // 2 * C * channel
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
        self.l, self.r = 1, 1
        self.Ls = {
            0: (0, 1),
            15: (1, 0.1),
            25: (10, 0.1),
            35: (10, 0.01),
        }

    def dist(self, ins, pos):
        return torch.norm(ins - pos, dim=1)

    def forward(self, x, lens, dists, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        anchor, positive, negative = x
        anchor_len, pos_len, neg_len = (d.type(torch.float32) for d in lens)
        pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

        anchor_embed_norm = torch.norm(anchor, dim=1)
        pos_embed_norm = torch.norm(positive, dim=1)
        neg_embed_norm = torch.norm(negative, dim=1)

        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        threshold = neg_dist - pos_dist
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)
        norm_loss = (anchor_embed_norm - anchor_len) ** 2 + \
                    (pos_embed_norm - pos_len) ** 2 + \
                    (neg_embed_norm - neg_len) ** 2
        mse_loss = (pos_embed_dist - pos_dist) ** 2 + \
                   (neg_embed_dist - neg_dist) ** 2 + \
                   (pos_neg_embed_dist - pos_neg_dist) ** 2

        return torch.mean(rank_loss), \
               torch.mean(mse_loss), \
               torch.mean(self.l * rank_loss +
                          self.r * (0.001 * norm_loss + mse_loss))
