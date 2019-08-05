import math
import itertools
import multiprocessing
import numpy as np
from main import intersect
from datasets import readlines, ivecs_read


dataset = "enron"


def ranking_recalls(sort, gt):
    ks = [1, 5, 10, 20, 50, 100, 1000]
    Ts = [2 ** i for i in range(2 + int(math.log2(len(sort[0]))))]
    print(" Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for t in Ts:
        print("%6d \t %6d \t" % (t, len(sort[0, :t])), end="")
        for top_k in ks:
            rc = intersect(gt[:, :top_k], sort[:, :t])
            print("%.4f \t" % (rc / float(top_k)), end="")
        print()


def hamming_similarity(args):
    str1, str2 = args
    return len(list(filter(
        lambda x: x[0] == x[1], zip(str1, str2))))


def distance(xq, xb):

    similarity = multiprocessing.Pool().map(
        hamming_similarity, itertools.product(xq, xb))
    return - np.array(similarity).reshape((len(xq), len(xb)))

def hamming_caller():
    print("# loading data")
    xq = readlines("data/%s/QUERY_cgk.txt" % dataset)
    xb = readlines("data/%s/BASE_cgk.txt" % dataset)
    gt = ivecs_read("data/%s/gt.ivecs" % dataset)[:100]
    print("# calculation distances")
    dist = distance(xq, xb)
    print("# sorting")
    sort = np.argsort(dist)
    ranking_recalls(sort, gt)


if __name__ == '__main__':
    hamming_caller()
