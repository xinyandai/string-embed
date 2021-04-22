import math
import time
import numpy as np

from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def _cgk(parameters):
    x, (h, M, C) = parameters
    i = 0
    j = 0
    out = np.empty(3 * M, np.int)
    out.fill(C)
    while j < 3 * M and i < len(x):
        out[j] = x[i]
        i += h[j][x[i]]
        j += 1
    return out


def cgk_string(h, strings, M, C):
    with Pool(cpu_count()) as pool:
        start_time = time.time()
        jobs = pool.imap(_cgk, zip(strings, [(h, M, C) for _ in strings]))
        cgk_list = list(tqdm(jobs, total=len(strings), desc="# CGK embedding"))
        print("# CGK embedding time: {}".format(time.time() - start_time))
        return np.array(cgk_list)


def random_seed(maxl, sig):
    return np.random.randint(low=0, high=2, size=(3 * maxl, sig))


def intersect(gs, ids):
    rc = np.mean([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])
    return rc


def ranking_recalls(sort, gt):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(len(sort[0]))))]
    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        print("%6d \t %6d \t" % (t, len(sort[0, :t])), end="")
        for top_k in ks:
            rc = intersect(gt[:, :top_k], sort[:, :t])
            print("%.4f \t" % (rc / float(top_k)), end="")
        print()


def hamming_distance(args):
    a, b = args
    return np.count_nonzero(a != b, axis=1)


def distance(xq, xb):
    def _distance(xq, xb):
        start_time = time.time()
        jobs = Pool().imap(hamming_distance, zip(xq, [xb for _ in xq]))
        dist = list(tqdm(jobs, total=len(xq), desc="# hamming counting"))
        print("# CGK hamming distance time: {}".format(time.time() - start_time))
        return np.array(dist).reshape((len(xq), len(xb)))

    if len(xq) < len(xb):
        return _distance(xb, xq).T
    else:
        return _distance(xq, xb)


def cgk_embedding(args, datahandler):
    h = random_seed(datahandler.M, datahandler.C)

    xq = cgk_string(h, datahandler.xq.sig, datahandler.M, datahandler.C)
    xb = cgk_string(h, datahandler.xb.sig, datahandler.M, datahandler.C)

    dist = distance(xq, xb)
    sort = np.argsort(dist)
    ranking_recalls(sort, datahandler.query_knn)
