import sys
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


def word2vec(file, max_length=None, C=ord('~') - ord(' ') + 2, first=' '):
    """
    :param file: the path to the file
    :return: 2-dimensional numpy array, of which each row denotes one string's one-hot coding
    """
    lines = readlines(file)

    lens = [len(line) for line in lines]
    if max_length is None:
        max_length = np.max(lens)
        if max_length % 2 != 0 :
            max_length += 1
    elif max_length < np.max(lens):
        warnings.warn("K is {} while strings may "
                      "exceed the maximum length {}"
                      .format(max_length, np.max(lens)))

    x = [[ord(c)-ord(first) + 1 for c in line]
         for line in lines]
    return OneHotString(C, max_length, x, lines), lens


def ivecs_read(file):
    """
    :param file:  the path to the ivecs file
    :return: 2 dimensional numpy array
    """
    a = np.fromfile(file, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


class OneHotString(Dataset):
    def __init__(self, C, M, ascii, lines):
        self.C, self.M = C, M
        self.ascii = ascii
        self.lines = lines

    def __getitem__(self, index):
        encode = np.zeros((self.C, self.M), dtype=np.float32)
        encode[np.array(self.ascii[index]),
               np.arange(len(self.ascii[index]))] = 1.0
        return torch.from_numpy(encode)

    def __len__(self):
        return len(self.ascii)


class TripletString(Dataset):
    def __init__(self, strings, lens, knn, K=50):
        self.strings, self.lens, self.knn = strings, lens, knn
        self.N, self.C, self.M = len(strings), strings.C, strings.M
        self.N, self.K = self.knn.shape
        self.K = min(K, self.K)
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


class PairwiseString(Dataset):
    pass