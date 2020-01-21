import math
import faiss
import argparse
import numpy as np

from utils import l2_dist, intersect
from embed_cnn import test_recall


def ss(xq, xb, G):
    n, d = xb.shape
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(n)))]


    nlist = 100
    m = 8
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                      # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(xb)
    index.add(xb)
    
    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        index.nprobe = t  # make comparable with experiment above
        D, ids = index.search(xq, t)  # search
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        for top_k in ks:
            rc = intersect(G[:, :top_k], ids)
            print("%.4f \t" % (rc / float(top_k)), end="")
        print()


def topk(xq, xb, xt, query_dist, train_dist):
    query_knn_ = np.argsort(query_dist)
    test_recall(xb, xq, query_knn_)
    ss(xq, xb, query_knn_)


def linear_fit(x, y, deg=1):
    x = x.reshape(-1)
    y = y.reshape(-1)
    indices = np.argwhere(~np.isnan(x)).reshape(-1)
    weights = np.polyfit(x[indices], y[indices], deg=deg)
    # poly1d_fn = np.poly1d(weights)
    print(weights)
    poly1d_fn = np.poly1d(weights)
    return poly1d_fn

def analyze(q, x, ed):
    l2 = l2_dist(q, x)

    from matplotlib import pyplot as plt

    idx = np.random.choice(np.size(ed), 1000)
    plt.scatter(ed.reshape(-1)[idx], l2.reshape(-1)[idx], color="r")
    plt.show()

# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

def ann(xq, xb, xt, query_dist, train_dist, args):
    # analyze(xt, xt, train_dist)
    # analyze(xq, xb, query_dist)
    bias = 0.0
    scales = 2.0**(np.arange(-10, 20))
    if args.dataset == "gen50ks.txt" and args.embed == 'cnn':
        scales =  np.linspace(0.01, 2.0, num=50)
    if args.dataset == "gen50ks.txt" and args.embed == 'gru':
        scales =  np.linspace(0, 4.0, num=50)
    if args.dataset == "trec" and args.embed == 'cnn':
        scales =  np.linspace(0, 2.0, num=50)
    if args.dataset == "trec" and args.embed == 'gru':
        scales =  np.linspace(2.5, 3.1, num=50)
        bias = 60
    if args.dataset == "enron" and args.embed == 'cnn':
        scales =  np.linspace(0., 1.0, num=50)
    if args.dataset == "enron" and args.embed == 'gru':
        scales =  np.linspace(0., 2.000, num=50)
    if args.dataset == "dblp" and args.embed == 'cnn':
        scales =  np.linspace(0.1, 2.0, num=50)
    if args.dataset == "dblp" and args.embed == 'gru':
        scales =  np.linspace(0.5, 1.6, num=50)
    if args.dataset == "uniref" and args.embed == 'cnn':
        scales =  np.linspace(0.5, 4.0, num=50)
    if args.dataset == "uniref" and args.embed == 'gru':
        scales =  np.linspace(0., 1.4, num=50)

    print(scales)
    thresholds = [1, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 300, 500, 800, 1000, 2000]
    train_dist_l2 = l2_dist(xt, xt)
    query_dist_l2 = l2_dist(xq, xb)
    threshold2dist = linear_fit(train_dist, train_dist_l2)
    print("thres\t l2thres\t", end='')
    for scale in scales:
        print("%2.3f\t" % scale, end='')
    print()

    for threshold in thresholds:
        gt = [np.argwhere(dist <= threshold) for dist in query_dist]
        threshold_l2 = threshold2dist(threshold)
        print("%6d\t %.6f\t" % (threshold, threshold_l2), end='')
        for scale in scales:
            items = [np.argwhere(dist <= bias + threshold_l2 * scale) for dist in query_dist_l2]
            recall = np.mean([len(np.intersect1d(i, j)) / len(i) for i, j in zip(gt, items) if len(i) > 0])
            print("%.3f\t" % (recall), end='')
        print()
    for threshold in thresholds:
        gt = [np.argwhere(dist <= threshold) for dist in query_dist]
        threshold_l2 = threshold2dist(threshold)
        print("%6d\t %.6f\t" % (threshold, threshold_l2), end='')
        for scale in scales:
            items = [np.argwhere(dist <= threshold_l2 * scale) for dist in query_dist_l2]
            precs = np.mean([len(np.intersect1d(i, j)) / len(j) if len(j) > 0 else 0 for i, j in zip(gt, items) if len(i) > 0])
            print("%.3f\t" % (precs), end='')
        print()

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


def load_vec(args):
    if args.embed == "cnn":
        data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
            args.shuffle_seed,
            args.embed,
            args.dataset,
            args.nt,
            args.nq,
            "" if args.maxl == 0 else "maxl{}".format(args.maxl),
        )
    else:
        data_file = "../ICLRcode/model/{}/{}/nt{}_nq{}{}".format(
            args.shuffle_seed,
            args.dataset,
            args.nt,
            args.nq,
            "" if args.maxl == 0 else "maxl{}".format(args.maxl),
        )

    print("# loading embeddings")
    if args.embed == 'gru':
        xb = np.load("{}/embedding_xb_0.npy".format(data_file))
    else:
        xb = np.load("{}/embedding_xb.npy".format(data_file))

    xt = np.load("{}/embedding_xt.npy".format(data_file))
    xq = np.load("{}/embedding_xq.npy".format(data_file))
    print("# ", xb.shape, xt.shape, xq.shape)
    data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
        args.shuffle_seed,
        'cnn',
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
    if args.embed == 'gru':
        # TODO bugs to fix
        xq, xt = xt, xq

    return xq, xb, xt, train_dist, query_dist


if __name__ == "__main__":
    args = get_args()
    xq, xb, xt, train_dist, query_dist = load_vec(args)
    print(args)
    query_dist = query_dist[:, :args.nb]
    xb = xb[:args.nb, :]
    topk(xq, xb, xt, query_dist, train_dist)
    # ann(xq, xb, xt, query_dist, train_dist, args)
