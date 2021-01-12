import os
import time
import tqdm
import random
import string
import argparse
import numpy as np

from multiprocessing import cpu_count

from utils import l2_dist
from embed_cnn import cnn_embedding
from embed_cgk import cgk_embedding
from k_medoids import k_medoids_embedding
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
            self.lines = [l[: self.maxl] for l in self.lines]
        self.ni = len(self.lines)
        self.nb = self.ni - self.nq - self.nt

        start_time = time.time()
        self.C, self.M, self.char_ids, self.alphabet = word2sig(self.lines, max_length=None)
        print("# Loading time: {}".format(time.time() - start_time))

        self.load_ids()
        self.load_dist()

        self.string_t = [self.lines[i] for i in self.train_ids]
        self.string_q = [self.lines[i] for i in self.query_ids]
        self.string_b = [self.lines[i] for i in self.base_ids]

        self.xt = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.train_ids]
        )
        self.xq = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.query_ids]
        )
        self.xb = StringDataset(
            self.C, self.M, [self.char_ids[i] for i in self.base_ids]
        )

    def random_str(self, M):
        """Generate a random string with a maximum length """
        return ''.join(random.choice(self.alphabet)
                       for _ in range(random.randint(1, M)))

    def random_trains(self, replace):
        root_dir = "folder/{}_{}/{}/{}/{}/".format(
            self.dataset, self.args.maxl, self.args.shuffle_seed,
            "random" if replace else "append", self.args.nr
        )

        os.makedirs(root_dir, exist_ok=True)
        random_text = root_dir + "random.txt"
        if not os.path.isfile(random_text):
            print("# generate random training samples " + random_text)
            self.train_rnd = [self.random_str(self.M) for _ in range(self.args.nr)]
            if not replace:
                print("# appended to training samples " + random_text)
                self.train_rnd =  [self.lines[i] for i in self.train_ids] + self.train_rnd
            with open(random_text, "w") as w:
                w.writelines("%s\n" % line for line in self.train_rnd)
            self.train_dist, self.train_knn = get_dist_knn(self.train_rnd)
            np.save(root_dir + "random_train_dist.npy", self.train_dist)
            np.save(root_dir + "random_train_knn.npy", self.train_knn)
        else:
            print("# loading random training samples " + random_text)
            self.train_rnd = readlines(random_text.format(self.args.dataset))
            self.train_dist = np.load(root_dir + "random_train_dist.npy")
            self.train_knn = np.load(root_dir + "random_train_knn.npy")

        _, _, train_sig, alphabet = word2sig(lines=self.train_rnd, max_length=self.M)
        self.xt = StringDataset(self.C, self.M, train_sig)

    def save_split(self):
        root_dir = "folder/{}_{}/{}/".format(
            self.dataset, self.args.maxl, self.args.shuffle_seed
        )
        os.makedirs(root_dir, exist_ok=True)
        with open(root_dir + "query", "w") as w:
            w.writelines("%s\n" % self.lines[i] for i in self.query_ids)
        with open(root_dir + "train", "w") as w:
            w.writelines("%s\n" % self.lines[i] for i in self.train_ids)
        with open(root_dir + "base", "w") as w:
            w.writelines("%s\n" % self.lines[i] for i in self.base_ids)

    def generate_ids(self):
        np.random.seed(self.args.shuffle_seed)
        idx = np.arange(self.ni)
        np.random.shuffle(idx)
        print("# shuffled index: ", idx)
        self.train_ids = idx[: self.nt]
        self.query_ids = idx[self.nt : self.nq + self.nt]
        self.base_ids = idx[self.nq + self.nt : self.nq + self.nt + self.nb]

    def generate_dist(self):
        self.train_dist, self.train_knn = get_dist_knn(
            [self.lines[i] for i in self.train_ids]
        )
        self.query_dist, self.query_knn = get_dist_knn(
            [self.lines[i] for i in self.query_ids],
            [self.lines[i] for i in self.base_ids],
        )

    def load_ids(self):
        idx_dir = "{}/".format(self.data_f)
        if not os.path.isfile(idx_dir + "train_idx.npy"):
            self.generate_ids()
            np.save(idx_dir + "train_idx.npy", self.train_ids)
            np.save(idx_dir + "query_idx.npy", self.query_ids)
            np.save(idx_dir + "base_idx.npy", self.base_ids)
        else:
            print("# loading indices from file")
            self.train_ids = np.load(idx_dir + "train_idx.npy")
            self.query_ids = np.load(idx_dir + "query_idx.npy")
            self.base_ids = np.load(idx_dir + "base_idx.npy")

        print(
            "# Unique signature     : {}".format(self.C),
            "# Maximum length       : {}".format(self.M),
            "# Sampled Train Items  : {}".format(self.nt),
            "# Sampled Query Items  : {}".format(self.nq),
            "# Number of Base Items : {}".format(self.nb),
            "# Number of Items      : {}".format(self.ni),
            sep="\n",
        )

    def load_dist(self):
        idx_dir = "{}/".format(self.data_f)
        if not os.path.isfile(idx_dir + "train_dist.npy"):
            self.generate_dist()
            np.save(idx_dir + "train_dist.npy", self.train_dist)
            np.save(idx_dir + "train_knn.npy", self.train_knn)
            np.save(idx_dir + "query_dist.npy", self.query_dist)
            np.save(idx_dir + "query_knn.npy", self.query_knn)
        else:
            print("# loading dist and knn from file")
            self.train_dist = np.load(idx_dir + "train_dist.npy")
            self.train_knn = np.load(idx_dir + "train_knn.npy")
            self.query_dist = np.load(idx_dir + "query_dist.npy")
            self.query_knn = np.load(idx_dir + "query_knn.npy")
            print("# train dist : {}".format(self.train_knn.shape))
            print("# query dist : {}".format(self.query_knn.shape))

    def set_nb(self, nb):
        if nb < len(self.base_ids):
            self.base_ids = self.base_ids[:nb]
            self.query_dist = self.query_dist[:, :nb]
            self.query_knn = get_knn(self.query_dist)
            self.xb.sig = self.xb.sig[:nb]


def analyze(q, x, ed):
    l2 = l2_dist(q, x)

    from matplotlib import pyplot as plt

    idx = np.random.choice(np.size(ed), 1000)
    plt.scatter(ed.reshape(-1)[idx], l2.reshape(-1)[idx], color="r")
    plt.title(args.dataset)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for String Embedding")

    parser.add_argument("--dataset", type=str, default=None, help="dataset")
    parser.add_argument("--nt", type=int, default=1000, help="# of training samples")
    parser.add_argument("--nr", type=int, default=1000, help="# of generated training samples")
    parser.add_argument("--nq", type=int, default=1000, help="# of query items")
    parser.add_argument("--nb", type=int, default=1385451, help="# of base items")
    parser.add_argument("--k", type=int, default=100, help="# sampling threshold")
    parser.add_argument("--epochs", type=int, default=4, help="# of epochs")
    parser.add_argument(
        "--shuffle-seed", type=int, default=808, help="seed for shuffle"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for sgd")
    parser.add_argument(
        "--test-batch-size", type=int, default=1024, help="batch size for test"
    )
    parser.add_argument("--channel", type=int, default=8, help="# of channels")
    parser.add_argument(
        "--mtc",  action="store_true", default=False, help="does we use multi channel as for input"
    )
    parser.add_argument("--embed-dim", type=int, default=128, help="output dimension")
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="save cnn model"
    )
    parser.add_argument(
        "--save-split", action="store_true", default=False, help="save split data folder"
    )
    parser.add_argument(
        "--save-embed", action="store_true", default=False, help="save embedding"
    )
    parser.add_argument(
        "--random-train", action="store_true", default=False, help="generate random training samples and replace"
    )
    parser.add_argument(
        "--random-append-train", action="store_true", default=False, help="generate random training samples and append"
    )

    parser.add_argument(
        "--embed-dir", type=str, default="", help="embedding save location"
    )
    parser.add_argument(
        "--recall", action="store_true", default=False, help="print recall"
    )
    parser.add_argument(
        "--bert", action="store_true", default=False, help="using bert or not"
    )
    parser.add_argument("--embed", type=str, default="cnn", help="embedding method")
    parser.add_argument("--maxl", type=int, default=0, help="max length of strings")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables GPU training"
    )
    args = parser.parse_args()
    data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
        args.shuffle_seed,
        args.embed if args.embed != "cgk" else "cnn",
        args.dataset,
        args.nt,
        args.nq,
        "" if args.maxl == 0 else "maxl{}".format(args.maxl),
    )
    os.makedirs(data_file, exist_ok=True)

    h = DataHandler(args, data_file)
    h.set_nb(args.nb)
    if args.save_split:
        h.save_split()
    if args.random_append_train:
        h.random_trains(replace=False)
    elif args.random_train:
        h.random_trains(replace=True)
    return args, h, data_file


def run_from_train(args, h, data_file):

    if args.embed == "cnn":
        xq, xb, xt = cnn_embedding(args, h, data_file)
        # analyze(xt, xt, h.train_dist)
        # analyze(xq, xb, h.query_dist)
    elif args.embed == "cgk":
        cgk_embedding(args, h)
    elif args.embed == "km":
        k_medoids_embedding(args, h)
    else:
        assert False


if __name__ == "__main__":
    args, h, data_file = get_args()
    run_from_train(args, h, data_file)
