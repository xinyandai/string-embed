import torch
import numpy as np
from random import randint
from torch.utils.data import Dataset

def readlines(file):
    """
    :param file: the path to the file
    :return: a list of string
    """
    lines = open(file, 'rb').read().splitlines()
    return [line.decode("utf8", "ignore") for line in lines]

def word2vec(file):
    """
    :param file: the path to the file
    :return: 2-dimensional numpy array, of which each row denotes one string's one-hot coding
    """
    lines = readlines(file)
    lens = [len(line) for line in lines]
    K = np.max(lens)
    if K % 2 != 0 :
        K += 1
    x = np.zeros((len(lines), 256, K), np.float32)
    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            x[i, ord(c), j] = 1
    return x.reshape((len(lines), 256 * K)), lens

def ivecs_read(file):
    """
    :param file:  the path to the ivecs file
    :return: 2 dimensional numpy array
    """
    a = np.fromfile(file, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


class TripletString(Dataset):
    def __init__(self, words, knn):
        self.strings, self.lens = word2vec(words)
        self.knn = ivecs_read(knn)
        self.N, self.D = self.strings.shape
        self.N, self.K = self.knn.shape
        self.strings = torch.from_numpy(self.strings[:self.N, :])

        self.K = 50
        self.counter = 0
        self.index = np.arange(self.N)

    def _shuffle(self):
        np.random.shuffle(self.index)

    def __getitem__(self, idx):
        if self.counter == self.N:
            self.counter = 0
            self._shuffle()
        self.counter += 1
        anchor = self.index[idx]
        positive = self.knn[anchor, randint(0, self.K-1)]
        negative = self.knn[anchor, randint(self.K, self.N-1)]
        return self.strings[anchor], \
               self.strings[positive], \
               self.strings[negative]

    def __len__(self):
        return self.N

class PaiswiseString(Dataset):
    pass