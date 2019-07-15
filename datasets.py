import warnings
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


def word2vec(file, K=None):
    """
    :param file: the path to the file
    :return: 2-dimensional numpy array, of which each row denotes one string's one-hot coding
    """
    lines = readlines(file)

    lens = [len(line) for line in lines]
    if K is None:
        K = np.max(lens)
        if K % 2 != 0 :
            K += 1
    elif K < np.max(lens):
        warnings.warn("K is {} while strings may "
                      "exceed the maximum length {}"
                      .format(K, np.max(lens)))

    x = np.zeros((len(lines), 26, K), np.float32)
    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            if j < K:
                x[i, ord(c)-ord('a'), j] = 1

    return x, lens


def ivecs_read(file):
    """
    :param file:  the path to the ivecs file
    :return: 2 dimensional numpy array
    """
    a = np.fromfile(file, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


class TripletString(Dataset):
    def __init__(self, strings, lens, knn, K=50):
        self.strings, self.lens, self.knn = strings, lens, knn
        self.N, self.C, self.M = self.strings.shape
        self.N, self.K = self.knn.shape
        self.strings = torch.from_numpy(self.strings)

        self.K = K
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
        return self.strings[anchor:anchor+1], \
               self.strings[positive:positive+1], \
               self.strings[negative:negative+1]

    def __len__(self):
        return self.N


class PaiswiseString(Dataset):
    pass