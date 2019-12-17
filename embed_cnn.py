import os
import time
import tqdm
import math
import torch
import numpy as np
import numba as nb

from trainer import train_epoch
from sorter import parallel_sort
from datasets import TripletString, StringDataset


def intersect(gs, ids):
    rc = np.mean([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])
    return rc


def test_recall(X, Q, G):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(len(X))))]

    sort_idx = parallel_sort("euclid", X, Q)

    print("# Probed \t Items \t", end="")
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
def _batch_embed(args, net, vecs: StringDataset, device):
    embedding = np.empty(shape=(len(vecs), args.embed_dim))
    for i in tqdm.tqdm(nb.prange(len(vecs)), desc="# batch embedding"):
        embedding[i, :] = net(vecs[i].to(device)).cpu().data.numpy()
    return embedding


def cnn_embedding(args, h):
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_loader = TripletString(h.xt, h.nt, h.train_knn, h.train_dist, K=50)

    model_file = "data/{}_nt{}_epoch{}_model.torch".format(
        args.dataset, args.nt, args.epochs
    )
    if os.path.isfile(model_file):
        model = torch.load(model_file)
    else:
        start_time = time.time()
        model = train_epoch(args, train_loader, device)
        if args.save_model:
            torch.save(model, model_file)
        train_time = time.time() - start_time
        print("# Training time: " + str(train_time) + "\n")
    model.eval()

    start_time = time.time()

    item = _batch_embed(args, model.embedding_net, h.xb, device)
    query = _batch_embed(args, model.embedding_net, h.xq, device)
    embed_time = time.time() - start_time
    print("# Embedding time: " + str(embed_time) + "\n")

    test_recall(item, query, h.query_knn)
