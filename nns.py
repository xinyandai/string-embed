import argparse
import numpy as np
from embed_cnn import test_recall

def topk(xq, xb, xt, query_knn_, query_dist, train_knn_, train_dist):
    test_recall(xb, xq[:100], query_knn_[:100, :])

def ann(xq, xb, xt, query_knn_, query_dist, train_knn_, train_dist):
    pass

def load_vec():
    parser = argparse.ArgumentParser(description="HyperParameters for String Embedding")

    parser.add_argument("--dataset", type=str, default="word", help="dataset")
    parser.add_argument("--nt", type=int, default=1000, help="# of training samples")
    parser.add_argument("--nq", type=int, default=1000, help="# of query items")
    parser.add_argument("--shuffle-seed", type=int, default=808, help="seed for shuffle")

    parser.add_argument("--recall", action="store_true", default=False, help="print recall")
    parser.add_argument("--embed", type=str, default="cnn", help="embedding method")
    parser.add_argument("--maxl", type=int, default=0, help="max length of strings")
    args = parser.parse_args()
    data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
        args.shuffle_seed,
        args.embed,
        args.dataset,
        args.nt,
        args.nq,
        "" if args.maxl == 0 else "_maxl{}".format(args.maxl),
    )

    print("# loading embeddings")
    xb = np.load("{}/embedding_xb.npy".format(data_file))
    xt = np.load("{}/embedding_xt.npy".format(data_file))
    xq = np.load("{}/embedding_xq.npy".format(data_file))

    data_file = "model/{}/{}/{}/nt{}_nq{}{}".format(
        args.shuffle_seed,
        'cnn',
        args.dataset,
        args.nt,
        args.nq,
        "" if args.maxl == 0 else "_maxl{}".format(args.maxl),
    )
    print("# loading distances")
    train_dist = np.load(data_file + '/train_dist.npy')
    train_knn_ = np.load(data_file + '/train_knn.npy')
    query_dist = np.load(data_file + '/query_dist.npy')
    query_knn_ = np.load(data_file + '/query_knn.npy')
    if args.embed == 'gru':
        # TODO bugs to fix
        xq, xt = xt, xq

    topk(xq, xb, xt, query_knn_, query_dist, train_knn_, train_dist)
    ann(xq, xb, xt, query_knn_, query_dist, train_knn_, train_dist)


if __name__ == "__main__":
    load_vec()