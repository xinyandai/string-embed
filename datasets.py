import time
import torch
import struct
import warnings
import Levenshtein
import numpy as np
from tqdm import tqdm
from random import randint
from multiprocessing import Pool
from torch.utils.data import Dataset


def f(x):
    a, B = x
    return [Levenshtein.distance(a, b) for b in B]


def all_pair_distance(A, B, n_thread):
    def all_pair(A, B, n_thread):
        with Pool(n_thread) as pool:
            start_time = time.time()
            edit = list(
                tqdm(
                    pool.imap(f, zip(A, [B for _ in A])),
                    total=len(A),
                    desc="# edit distance {}x{}".format(len(A), len(B)),
                )
            )
            print("# Calculate edit distance time: {}".format(time.time() - start_time))
            return np.array(edit)

    if len(A) < len(B):
        return all_pair(B, A, n_thread).T
    else:
        return all_pair(A, B, n_thread)


def readlines(file):
    """
    :param file: the path to the file
    :return: a list of string
    """
    lines = open(file, "rb").read().splitlines()
    return [line.decode("utf8", "ignore") for line in lines]


def word2sig(lines, max_length=None):
    """
    :param file: the path to the file
    :return: 2-dimensional numpy array, of which each row denotes one string's one-hot coding
    """
    lens = [len(line) for line in lines]
    if max_length is None:
        max_length = np.max(lens)
        if max_length % 2 != 0:
            max_length += 1
    elif max_length < np.max(lens):
        warnings.warn(
            "K is {} while strings may "
            "exceed the maximum length {}".format(max_length, np.max(lens))
        )

    all_chars = dict()
    all_chars["counter"] = 0

    def to_ord(c):
        nonlocal all_chars
        if not (c in all_chars):
            all_chars[c] = all_chars["counter"]
            all_chars["counter"] = all_chars["counter"] + 1
        return all_chars[c]

    x = [[to_ord(c) for c in line] for line in lines]

    return all_chars["counter"], max_length, x


def ivecs_read(file):
    """
    :param file:  the path to the ivecs file
    :return: 2 dimensional numpy array
    """
    a = np.fromfile(file, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view("float32")


def fvecs_writer(filename, vecs):
    f = open(filename, "ab")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack("i" * len(dimension), *dimension))
        f.write(struct.pack("f" * len(x), *x))

    f.close()


class StringDataset(Dataset):
    def __init__(self, C, M, sig):
        self.C, self.M = C, M
        self.sig = sig

    def __getitem__(self, index):
        encode = np.zeros((self.C, self.M), dtype=np.float32)
        encode[np.array(self.sig[index]), np.arange(len(self.sig[index]))] = 1.0
        return torch.from_numpy(encode)

    def __len__(self):
        return len(self.sig)


class TripletString(Dataset):
    def __init__(self, strings, lens, knn, dist, K):

        self.lens, self.knn, self.dist = lens, knn, dist
        self.N, self.C, self.M = len(strings), strings.C, strings.M
        self.N, self.K = self.knn.shape
        self.K = min(K, self.K)
        self.strings = strings
        self.counter = 0
        self.index = np.arange(self.N)
        self.avg_dist = np.mean(self.dist)

    def _shuffle(self):
        np.random.shuffle(self.index)

    def __getitem__(self, idx):
        if self.counter == self.N:
            self.counter = 0
            self._shuffle()
        self.counter += 1
        anchor = self.index[idx]
        positive = self.knn[anchor, randint(0, min(self.N - 1, self.K * 2))]
        negative = self.knn[anchor, randint(0, min(self.N - 1, self.K * 2))]
        pos_dist = self.dist[anchor, positive]
        neg_dist = self.dist[anchor, negative]
        if pos_dist > neg_dist:
            positive, negative = negative, positive
            pos_dist, neg_dist = neg_dist, pos_dist
        pos_neg_dist = self.dist[positive, negative]

        return (
            (self.strings[anchor], self.strings[positive], self.strings[negative]),
            (i / self.avg_dist for i in (pos_dist, neg_dist, pos_neg_dist)),
        )

    def __len__(self):
        return self.N
