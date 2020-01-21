import os
import numpy as np
from main import get_args
from nns import linear_fit
from embed_cgk import random_seed, cgk_string, distance


threshold = 1000

args, data_handler, data_file = get_args()
train_dist, query_dist = data_handler.train_dist, data_handler.query_dist
train_idx = np.where(train_dist < threshold)
query_idx = np.where(query_dist < threshold)

dis_dir = "cgk_dist/{}".format(args.dataset)
os.makedirs(dis_dir, exist_ok=True)
if not os.path.isfile(dis_dir + "train_idx.npy"):
    h = random_seed(data_handler.M, data_handler.C)
    xq = cgk_string(h, data_handler.xq.sig, data_handler.M)
    xt = cgk_string(h, data_handler.xt.sig, data_handler.M)
    xb = cgk_string(h, data_handler.xb.sig, data_handler.M)

    train_dist_hm = distance(xt, xt)
    query_dist_hm = distance(xq, xb)

    np.save(dis_dir + "train_dist_hm.npy", train_dist_hm)
    np.save(dis_dir + "query_dist_hm.npy", query_dist_hm)
else:
    train_dist_hm = np.load(dis_dir + "train_dist_hm.npy")
    query_dist_hm = np.load(dis_dir + "query_dist_hm.npy")

l2ed_gru = linear_fit(
    train_dist_hm[train_idx],
    train_dist[train_idx], deg=2)
print(np.mean(np.abs(l2ed_gru(query_dist_hm[query_idx]) / query_dist[query_idx] - 1.0)))