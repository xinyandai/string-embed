import numpy as np
import numba as nb
import math
import tqdm

@nb.jit
def arg_sort(distances):
    top_k = min(131072, len(distances)-1)
    indices = np.argpartition(distances, top_k)[:top_k]
    return indices[np.argsort(distances[indices])]


@nb.jit
def product_arg_sort(q, compressed):
    distances = np.dot(compressed, -q)
    return arg_sort(distances)


@nb.jit
def angular_arg_sort(q, compressed, norms_sqr):
    norm_q = np.linalg.norm(q)
    distances = np.dot(compressed, q) / (norm_q * norms_sqr)
    return arg_sort(distances)


@nb.jit
def euclidean_arg_sort(q, compressed):
    distances = np.linalg.norm(q - compressed, axis=1)
    return arg_sort(distances)


@nb.jit
def sign_arg_sort(q, compressed):
    distances = np.empty(len(compressed), dtype=np.int32)
    for i in range(len(compressed)):
        distances[i] = np.count_nonzero((q > 0) != (compressed[i] > 0))
    return arg_sort(distances)


@nb.jit
def euclidean_norm_arg_sort(q, compressed, norms_sqr):
    distances = norms_sqr - 2.0 * np.dot(compressed, q)
    return arg_sort(distances)


@nb.jit
def parallel_sort(metric, compressed, Q, X, norms_sqr=None):
    """
    for each q in 'Q', sort the compressed items in 'compressed' by their distance,
    where distance is determined by 'metric'
    :param metric: euclid product
    :param compressed: compressed items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """

    rank = np.empty((Q.shape[0], min(131072, compressed.shape[0]-1)), dtype=np.int32)

    p_range = tqdm.tqdm(nb.prange(Q.shape[0]))

    if metric == 'product':
        for i in p_range:
            rank[i, :] = product_arg_sort(Q[i], compressed)
    elif metric == 'angular':
        if norms_sqr is None:
            norms_sqr = np.linalg.norm(compressed, axis=1) ** 2
        for i in p_range:
            rank[i, :] = angular_arg_sort(Q[i], compressed, norms_sqr)
    elif metric == 'euclid_norm':
        if norms_sqr is None:
            norms_sqr = np.linalg.norm(compressed, axis=1) ** 2
        for i in p_range:
            rank[i, :] = euclidean_norm_arg_sort(Q[i], compressed, norms_sqr)
    else:
        for i in p_range:
            rank[i, :] = euclidean_arg_sort(Q[i], compressed)

    return rank


@nb.jit
def true_positives(topK, Q, G, T):
    result = np.empty(shape=(len(Q)))
    for i in nb.prange(len(Q)):
        result[i] = len(np.intersect1d(G[i], topK[i][:T]))
    return result


class Sorter(object):
    def __init__(self, compressed, Q, X, metric, norms_sqr=None):
        self.Q = Q
        self.X = X

        self.topK = parallel_sort(metric, compressed, Q, X, norms_sqr=norms_sqr)

    def recall(self, G, T):
        t = min(T, len(self.topK[0]))
        return t, self.sum_recall(G, T) / len(self.Q)

    def sum_recall(self, G, T):
        assert len(self.Q) == len(self.topK), "number of query not equals"
        assert len(self.topK) <= len(G), "number of queries should not exceed the number of queries in ground truth"
        true_positive = true_positives(self.topK, self.Q, G, T)
        return np.sum(true_positive) / len(G[0])  # TP / K


class BatchSorter(object):
    def __init__(self, compressed, Q, X, G, Ts, metric, batch_size, norms_sqr=None):
        self.Q = Q
        self.X = X
        self.recalls = np.zeros(shape=(len(Ts)))
        for i in range(math.ceil(len(Q) / float(batch_size))):
            q = Q[i*batch_size: (i + 1) * batch_size, :]
            g = G[i*batch_size: (i + 1) * batch_size, :]
            sorter = Sorter(compressed, q, X, metric=metric, norms_sqr=norms_sqr)
            self.recalls[:] = self.recalls[:] + [sorter.sum_recall(g, t) for t in Ts]
        self.recalls = self.recalls / len(self.Q)

    def recall(self):
        return self.recalls