import math
import argparse
import numpy as np
from utils import arg_sort, intersect_sizes


def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for String Embedding")

    parser.add_argument("--dataset", type=str, default="gen50ks.txt", help="dataset")
    parser.add_argument("--nt", type=int, default=1000, help="# of training samples")
    parser.add_argument("--nq", type=int, default=1000, help="# of query items")
    parser.add_argument("--nb", type=int, default=1385451, help="# of base items")
    parser.add_argument("--shuffle-seed", type=int, default=808, help="seed for shuffle")

    parser.add_argument("--recall", action="store_true", default=False, help="print recall")
    parser.add_argument("--embed", type=str, default="cnn", help="embedding method")
    parser.add_argument("--maxl", type=int, default=0, help="max length of strings")
    args = parser.parse_args()
    return args

def load_dist(args):
    data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
        args.shuffle_seed,
        args.embed,
        args.dataset,
        args.nt,
        args.nq,
        "" if args.maxl == 0 else "maxl{}".format(args.maxl),
    )

    print("# loading distances")
    train_dist = np.load(data_file + '/train_dist.npy')
    print("# loaded train_dist")
    query_dist = np.load(data_file + '/query_dist.npy')
    print("# loaded query_dist")
    return train_dist, query_dist

def load_vec(args, embeding=""):
    data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
        args.shuffle_seed,
        args.embed,
        args.dataset,
        args.nt,
        args.nq,
        "" if args.maxl == 0 else "maxl{}".format(args.maxl),
    )
    print("# loading embeddings")
    def my_load(x):
        print("# loading from " + x)
        return np.load(x)
    xb = my_load("{}/{}embedding_xb.npy".format(data_file, embeding))
    xt = my_load("{}/{}embedding_xt.npy".format(data_file, embeding))
    xq = my_load("{}/{}embedding_xq.npy".format(data_file, embeding))
    print("# ", xb.shape, xt.shape, xq.shape)
    return [xq, xb, xt]

def multi_test_recall(Xs, G):
    """
    :param Xs: list fo [xq, xb, xt]
    :param G:
    :return:
    """
    print("# {}".format(np.shape(Xs)))
    nq = len(Xs[0][0])
    nb = len(Xs[0][1])
    nt = len(Xs[0][2])
    print("# nb {}, nq {}, nt {}".format(nb, nq, nt))
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(nb)))]
    for i in range(2):
        Ts = list(Ts) + list(np.convolve(Ts, [0.5, 0.5]).astype(np.int))
        Ts = np.unique(Ts)

    sort_idx = [arg_sort(xq, xb) for [xq, xb, xt] in Xs]

    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        indices = np.array([idx[:, :t] for idx in sort_idx])
        probed = [set(indices[:,q,:].flatten()) for q in range(nq)]
        avg_probe = np.mean([len(i) for i in probed])
        print("%6d \t %6d \t" % (t, avg_probe), end="")
        tps = [intersect_sizes(G[:, :top_k], probed) / float(top_k) for top_k in ks]
        rcs = [np.mean(t) for t in tps]
        # vrs = [np.std(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        # for vr in vrs:
        #     print("%.4f \t" % vr, end="")
        print()


if __name__ == "__main__":
    args = get_args()
    train_dist, query_dist = load_dist(args)
    query_dist = query_dist[:, :args.nb]
    query_knn_ = np.argsort(query_dist)
    Xs = [load_vec(args, "embed_"+str(i)+"/") for i in range(1, 2)]
    multi_test_recall(Xs, query_knn_)

