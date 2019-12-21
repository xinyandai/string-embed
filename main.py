import os
import time
import tqdm
import pickle
import argparse
import numpy as np

from multiprocessing import cpu_count

from embed_cnn import cnn_embedding
from embed_cgk import cgk_embedding
from datasets import readlines, word2sig, StringDataset, all_pair_distance


def get_knn(dist):
    knn = np.empty(dtype=np.int32, shape=(len(dist), len(dist[0])))
    for i in tqdm.tqdm(range(len(dist)), desc="# sorting for KNN indices"):
        knn[i, :] = np.argsort(dist[i, :])
    return knn


def get_dist_knn(queries, base=None):
    if base is None:
        base = queries

    dist = all_pair_distance(queries, base, cpu_count())
    return dist, get_knn(dist)


class DataHandler:
    def __init__(self, args, data_f):
        self.data_f = data_f
        self.args = args
        self.nt = args.nt
        self.nq = args.nq
        self.maxl = args.maxl
        self.dataset = args.dataset

        self.lines = readlines("data/{}".format(args.dataset))
        if self.maxl != 0:
            self.lines = [l[:self.maxl] for l in self.lines]
        self.ni = len(self.lines)
        self.nb = self.ni - self.nq - self.nt

        start_time = time.time()
        self.C, self.M, self.char_ids = word2sig(self.lines, max_length=None)
        print("# Loading time: {}".format(time.time() - start_time))

        self.load_ids()
        self.load_dist()

        self.xt = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.train_ids]
        )
        self.xq = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.query_ids]
        )
        self.xb = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.base_ids]
        )

    def generate_ids(self):
        np.random.seed(self.args.shuffle_seed)
        idx = np.arange(self.ni)
        np.random.shuffle(idx)
        print("# shuffled index: ", idx)
        self.train_ids = idx[: self.nt]
        self.query_ids = idx[self.nt: self.nq + self.nt]
        self.base_ids = idx[self.nq + self.nt:]

    def generate_dist(self):
        self.train_dist, self.train_knn = get_dist_knn(
            [self.lines[i] for i in self.train_ids]
        )
        self.query_dist, self.query_knn = get_dist_knn(
            [self.lines[i] for i in self.query_ids],
            [self.lines[i] for i in self.base_ids]
        )

    def load_ids(self):
        idx_dir = "{}/".format(self.data_f)
        if not os.path.isfile(idx_dir + 'train_idx.npy'):
            self.generate_ids()
            np.save(idx_dir + 'train_idx.npy', self.train_ids)
            np.save(idx_dir + 'query_idx.npy', self.query_ids)
            np.save(idx_dir + 'base_idx.npy', self.base_ids)
        else:
            print("# loading indices from file")
            self.train_ids = np.load(idx_dir + 'train_idx.npy')
            self.query_ids = np.load(idx_dir + 'query_idx.npy')
            self.base_ids = np.load(idx_dir + 'base_idx.npy')

        print(
            "# Unique signature     : {}".format(self.C),
            "# Maximum length       : {}".format(self.M),
            "# Sampled Train Items  : {}".format(self.nt),
            "# Sampled Query Items  : {}".format(self.nq),
            "# Number of Base Items : {}".format(self.nb),
            "# Number of Items      : {}".format(self.ni),
            sep='\n'
        )

    def load_dist(self):
        idx_dir = "{}/".format(self.data_f)
        if not os.path.isfile(idx_dir + 'train_dist.npy'):
            self.generate_dist()
            np.save(idx_dir + 'train_dist.npy', self.train_dist)
            np.save(idx_dir + 'train_knn.npy', self.train_knn)
            np.save(idx_dir + 'query_dist.npy', self.query_dist)
            np.save(idx_dir + 'query_knn.npy', self.query_knn)
        else:
            print("# loading dist and knn from file")
            self.train_dist = np.load(idx_dir + 'train_dist.npy')
            self.train_knn = np.load(idx_dir + 'train_knn.npy')
            self.query_dist = np.load(idx_dir + 'query_dist.npy')
            self.query_knn = np.load(idx_dir + 'query_knn.npy')

    def set_nb(self, nb):
        if nb >= len(self.base_ids):
            self.base_ids = self.base_ids[:nb]
            self.query_dist = self.query_dist[:, :nb]
            self.query_knn = get_knn(self.query_dist)
            self.xb.sig = self.xb.sig[:nb]

def analyze(q, x, ed):
    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = np.sqrt(sqr_q + sqr_x - 2 *(q @ x))

    from matplotlib import pyplot as plt
    idx = np.random.choice(np.size(ed), 1000)
    plt.scatter(ed.reshape(-1)[idx], l2.reshape(-1)[idx], color='r')
    plt.title(args.dataset)
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for String Embedding")

    parser.add_argument("--dataset", type=str, default=None, help="dataset")
    parser.add_argument("--nt", type=int, default=1000, help="# of training samples")
    parser.add_argument("--nq", type=int, default=1000, help="# of query items")
    parser.add_argument("--nb", type=int, default=1385451, help="# of base items")
    parser.add_argument("--k", type=int, default=100, help="# sampling threshold")
    parser.add_argument("--epochs", type=int, default=4, help="# of epochs")
    parser.add_argument("--shuffle-seed", type=int, default=808, help="seed for shuffle")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for sgd")
    parser.add_argument(
        "--test-batch-size", type=int, default=1024, help="batch size for test"
    )
    parser.add_argument("--channel", type=int, default=8, help="# of channels")
    parser.add_argument("--embed-dim", type=int, default=128, help="output dimension")
    parser.add_argument("--save-model", action="store_true", default=False, help="save cnn model")
    parser.add_argument("--recall", action="store_true", default=False, help="print recall")
    parser.add_argument("--embed", type=str, default="cnn", help="embedding method")
    parser.add_argument("--maxl", type=int, default=0, help="max length of strings")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables GPU training"
    )
    args = parser.parse_args()
    data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
        args.shuffle_seed,
        args.embed,
        args.dataset,
        args.nt,
        args.nq,
        "" if args.maxl == 0 else "_maxl{}".format(args.maxl),
    )
    os.makedirs(data_file, exist_ok=True)

    h = DataHandler(args, data_file)
    h.set_nb(args.nb)
    return args, h, data_file


def run_from_train(args, h, data_file):

    if args.embed == "cnn":
        cnn_embedding(args, h, data_file)
    elif args.embed == "cgk":
        cgk_embedding(args, h)
    else:
        assert False

if __name__ == "__main__":
    args, h, data_file = get_args()
    run_from_train(args, h, data_file)
