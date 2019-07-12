import math
import torch
import numpy as np
from trainer import train_epoch
from sorter import parallel_sort
from datasets import ivecs_read
from datasets import word2vec
from datasets import TripletString


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


def _batch_embed(net, vecs, device):
    chunk_size = 100
    embedding = np.empty(shape=(len(vecs), net.embedding))
    for i in range(math.ceil(len(vecs) / chunk_size)):
        sub = vecs[i * chunk_size: (i + 1) * chunk_size, :]
        embedding[i * chunk_size: (i + 1) * chunk_size, :] = \
            net(torch.from_numpy(sub).to(device)).cpu().data.numpy()
    return embedding


def main(epoch):
    train_knn = ivecs_read("data/word/knn.ivecs")
    K = 30
    xb, nb = word2vec("data/word/base.txt", K=K)
    xt, nt =  word2vec("data/word/train.txt", K=K)
    xq, nq =  word2vec("data/word/query.txt", K=K)[:100]
    gt = ivecs_read("data/word/gt.ivecs")[:100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = TripletString(xt, nt, train_knn)

    model = train_epoch(epoch, train_loader, device)

    item = _batch_embed(model.embedding_net, xb, device)
    query = _batch_embed(model.embedding_net, xq, device)

    test_recall(item, query, gt)

main(0)
