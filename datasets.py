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


def all_pair_distance(A, B, n_thread, progress=True):
    bar = tqdm if progress else lambda iterable, total, desc: iterable

    def all_pair(A, B, n_thread):
        with Pool(n_thread) as pool:
            start_time = time.time()
            edit = list(
                bar(
                    pool.imap(f, zip(A, [B for _ in A])),
                    total=len(A),
                    desc="# edit distance {}x{}".format(len(A), len(B)),
                ))
            if progress:
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
        warnings.warn("K is {} while strings may " "exceed the maximum length {}".format(max_length, np.max(lens)))

    all_chars = dict()
    all_chars["counter"] = 0
    alphabet = ''

    def to_ord(c):
        nonlocal all_chars
        nonlocal alphabet
        if not (c in all_chars):
            alphabet += c
            all_chars[c] = all_chars["counter"]
            all_chars["counter"] = all_chars["counter"] + 1
        return all_chars[c]

    x = [[to_ord(c) for c in line] for line in lines]

    return all_chars["counter"], max_length, x, alphabet


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
        self.bert_dataset = False
        self.bert_sig = None

    def __getitem__(self, index):
        if self.bert_dataset == False:
            encode = np.zeros((self.C, self.M), dtype=np.float32)
            encode[np.array(self.sig[index]), np.arange(len(self.sig[index]))] = 1.0
            return torch.from_numpy(encode)
        else:
            return self.bert_sig[index]

    def __len__(self):
        return len(self.sig)

    def to_original_dataset(self):
        self.bert_dataset = False

    def to_bert_dataset(self, char_alphabet):
        self.bert_dataset = True
        strs = []
        for word in self.sig:
            new_word = ""
            for idx in word:
                new_word += char_alphabet[idx]
            strs.append(new_word)

        self.bert_sig = strs


class TripletString(Dataset):

    def __init__(self, strings, lens, knn, dist, K):

        self.lens, self.knn, self.dist = lens, knn, dist
        self.N, self.C, self.M = len(strings), strings.C, strings.M
        self.N, self.K = self.knn.shape
        self.K = min(K, self.K)
        self.strings = strings
        self.index = np.arange(self.N)
        self.avg_dist = np.mean(self.dist)
        self.lens = [np.sum(s) for s in self.strings.sig]

    def __getitem__(self, idx):
        anchor = idx
        positive = self.knn[anchor, randint(1, min(self.N - 1, self.K * 2))]
        negative = self.knn[anchor, randint(1, min(self.N - 1, self.K * 2))]
        pos_dist = self.dist[anchor, positive]
        neg_dist = self.dist[anchor, negative]
        if pos_dist > neg_dist:
            positive, negative = negative, positive
            pos_dist, neg_dist = neg_dist, pos_dist
        pos_neg_dist = self.dist[positive, negative]

        return (
            self.strings[anchor],
            self.strings[positive],
            self.strings[negative],
            self.lens[anchor] / self.avg_dist,
            self.lens[positive] / self.avg_dist,
            self.lens[negative] / self.avg_dist,
            pos_dist / self.avg_dist,
            neg_dist / self.avg_dist,
            pos_neg_dist / self.avg_dist,
        )

    def __len__(self):
        return self.N
