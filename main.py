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
    def __init__(self, dataset, n_t, n_q):
        self.dataset = dataset
        self.nt = n_t
        self.nq = n_q

        lines = readlines("data/{}".format(dataset))
        self.ni = len(lines)
        self.nb = self.ni - self.nq - self.nt

        start_time = time.time()
        self.C, self.M, self.char_ids = word2sig(lines, max_length=None)
        print("# Loading time: {}".format(time.time() - start_time))
        idx = np.arange(self.ni)
        np.random.shuffle(idx)
        self.train_ids = idx[: self.nt]
        self.query_ids = idx[self.nt : self.nq + self.nt]
        self.base_ids = idx[self.nq + self.nt :]

        self.train_dist, self.train_knn = get_dist_knn(
            [lines[i] for i in self.train_ids]
        )
        self.query_dist, self.query_knn = get_dist_knn(
            [lines[i] for i in self.query_ids], [lines[i] for i in self.base_ids]
        )

        print(
            "# Unique signature     : {}\n".format(self.C),
            "# Maximum length       : {}\n".format(self.M),
            "# Sampled Train Items  : {}\n".format(self.nt),
            "# Sampled Query Items  : {}\n".format(self.nq),
            "# Number of Base Items : {}\n".format(self.nb),
            "# Number of Items : {}\n".format(self.ni),
        )

        self.xt = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.train_ids]
        )
        self.xq = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.query_ids]
        )
        self.xb = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.base_ids]
        )

    def set_nb(self, nb):
        if nb != len(self.base_ids):
            self.base_ids = self.base_ids[:nb]
            self.query_dist = self.query_dist[:, :nb]
            self.query_knn = get_knn(self.query_dist)
            self.xb.sig = self.xb.sig[:nb]


def run_from_train(args):
    data_file = "data/{}_nt{}_nq{}".format(args.dataset, args.nt, args.nq)

    if os.path.isfile(data_file):
        f = open(data_file, "rb")
        h = pickle.load(f)
    else:
        h = DataHandler(args.dataset, args.nt, args.nq)
        f = open(data_file, "wb")
        pickle.dump(h, f)
    h.set_nb(args.nb)
    if args.embed == "cnn":
        cnn_embedding(args, h)
    elif args.embed == "cgk":
        cgk_embedding(args, h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperParameters for String Embedding")

    parser.add_argument("--dataset", type=str, default=None, help="dataset")
    parser.add_argument(
        "--nt", type=int, default=1000, help="number of training samples"
    )
    parser.add_argument("--nq", type=int, default=100, help="number of query items")
    parser.add_argument("--nb", type=int, default=50000, help="number of query items")
    parser.add_argument(
        "--epochs", type=int, default=4, help="number of epochs to train"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for sgd")
    parser.add_argument(
        "--embed-dim", type=int, default=128, help="embedding dimension"
    )
    parser.add_argument("--save-model", type=bool, default=False, help="save cnn model")
    parser.add_argument("--embed", type=str, default="cnn", help="embedding method")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables GPU training"
    )
    args = parser.parse_args()
    run_from_train(args)
