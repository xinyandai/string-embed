import tqdm
import math
import torch
import argparse
import numpy as np
import numba as nb
from trainer import train_epoch
from sorter import parallel_sort
from datasets import ivecs_read, readlines, \
    word2vec, TripletString, OneHotString


def intersect(gs, ids):
    rc = np.mean([
        len(np.intersect1d(g, list(id)))
        for g, id in zip(gs, ids)])
    return rc


def test_recall(X, Q, G):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(len(X))))]

    sort_idx = parallel_sort("euclid", X, Q)

    print(" Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        ids = sort_idx[:, :t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        for top_k in ks:
            rc = intersect(G[:, :top_k], ids)
            print("%.4f \t" % (rc / float(top_k)), end="")
        print()

@nb.jit
def _batch_embed(args, net, vecs: OneHotString, device):
    embedding = np.empty(shape=(len(vecs), args.embed_dim))
    for i in tqdm.tqdm(nb.prange(len(vecs))):
        embedding[i, :] = net(vecs[i].to(device)).cpu().data.numpy()
    return embedding


dataset = "enron"
K = 65536
# dataset = "word"
# K = 32

def run_from_train(args):
    print("# loading data")
    train_knn = ivecs_read("data/%s/knn.ivecs" % dataset)
    xb, nb = word2vec("data/%s/base.txt" % dataset, max_length=K)
    xt, nt =  word2vec("data/%s/train.txt" % dataset, max_length=K)
    xq, nq = word2vec("data/%s/query.txt" % dataset, max_length=K)
    gt = ivecs_read("data/%s/gt.ivecs" % dataset)[:100]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda
                          else "cpu")
    train_loader = TripletString(xt, nt, train_knn)

    print("# training")
    model = train_epoch(args, train_loader, device)

    print("# embedding")
    item = _batch_embed(args, model.embedding_net, xb, device)
    query = _batch_embed(args, model.embedding_net, xq, device)

    test_recall(item, query, gt)


def statistic():
    lines = readlines("data/%s/%s" % (dataset, dataset))
    lens = list(map(len, lines))
    ords = np.hstack([[ord(c) for c in line] for line in lines])
    print(np.max(lens), np.min(lens), np.mean(lens))
    print(ords.shape)
    print(np.max(ords.reshape(-1)), np.min(ords.reshape(-1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='HyperParameters for String Embedding')
    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--embed-dim', type=int, default=32,
                        help='embedding dimension')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables GPU training')
    args = parser.parse_args()
    run_from_train(args)
