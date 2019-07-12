import tqdm
import numpy as np
import numba as nb


def get_top_k(X):
    """
    normal we do not have to sort the whole array
    :param X:
    :return:
    """
    return min(131072, len(X) - 1)


@nb.jit
def arg_sort(distances, top_k):
    indices = np.argpartition(distances, top_k)[:top_k]
    return indices[np.argsort(distances[indices])]


@nb.jit
def product_arg_sort(q, X, top_k):
    distances = np.dot(X, -q)
    return arg_sort(distances, top_k)


@nb.jit
def euclidean_arg_sort(q, X, top_k):
    distances = np.linalg.norm(q - X, axis=1)
    return arg_sort(distances, top_k)


@nb.jit
def parallel_sort(metric, X, Q):
    """
    :param metric: euclid product
    :param X: X items, same dimension as origin data, shape(N * D)
    :param Q: queries, shape(len(Q) * D)
    :return:
    """
    top_k = get_top_k(X)
    rank = np.empty((Q.shape[0], top_k), dtype=np.int32)

    p_range = tqdm.tqdm(nb.prange(Q.shape[0]))

    if metric == 'product':
        for i in p_range:
            rank[i, :] = product_arg_sort(Q[i], X, top_k)
    else:
        for i in p_range:
            rank[i, :] = euclidean_arg_sort(Q[i], X, top_k)
    return rank
