import random
import numpy as np
from tqdm import tqdm
from functools import partial
from Levenshtein import distance
from multiprocessing import Pool, cpu_count


def calculate_medoid(items):
    return min(items, key=partial(max_distance, items))


def total_distance(items, medoid):
    return sum(distance(medoid, o) for o in items)


def max_distance(items, medoid):
    return max(distance(medoid, o) for o in items)


def associate_(args_):
    o, medoids = args_
    medoid = min(medoids, key=lambda m: distance(m, o))
    return medoid


def associate(pool, medoids, items):
    assigned = list(
        pool.imap(associate_, [(x, medoids) for x in items])
    )
    clusters = {medoid: [] for medoid in medoids}
    for x, ci in zip(items, assigned):
        clusters[ci].append(x)
    return assigned, clusters


def cost(solution):
    return sum(
        max_distance(items, medoid)
        for medoid, items in solution.items()
    )


def clustered(items, n_clusters, iterations):
    """
    Implements K-Medoids clustering
    """
    with Pool(cpu_count()) as pool:
        # start off clustering around random medoids
        medoids = random.sample(items, n_clusters)

        assigned, clusters = associate(pool, medoids, items)

        for i in tqdm(range(iterations)):
            previous = clusters
            # update medoids
            medoids = list(
                pool.imap(calculate_medoid, clusters.values())
            )
            # recompute clusters with new medoids
            assigned, clusters = associate(pool, medoids, items)
            # has the solution converged?
            if cost(clusters) == cost(previous):
                break

    radius = {
        medoid: max_distance(items, medoid)
        for medoid, items in clusters.items()
    }
    distortion = {
        x: distance(x, ci) for x, ci in zip(items, assigned)
    }
    return clusters, radius, distortion


def search_radius(args_):
    q, clusters, radius, T = args_
    num_evaluated = 0
    for k, v in clusters.items():
        dist = distance(k, q)
        if dist - radius[k] < T:
            num_evaluated += len(v)
    return num_evaluated


def search_distortion(args_):
    q, clusters, distortion, T = args_
    num_evaluated = 0
    for k, xs in clusters.items():
        dist = distance(k, q)
        for x in xs:
            if dist - distortion[x] < T:
                num_evaluated += 1
    return num_evaluated


def k_medoids_embedding(args, h):
    clusters, radius, distortion = clustered(h.string_b, 1024 * 8, 10)
    radius_list = list(radius.values())
    for p in [0, 10, 20, 30, 50, 60, 70, 80, 90, 100]:
        print("p =", p, "%  : ", np.percentile(radius_list, p))

    for T in [1, 2, 5, 10]:
        with Pool(cpu_count()) as pool:
            search_res = list(
                pool.imap(
                    search_radius,
                    ((q, clusters, radius, T) for q in h.string_q),
                )
            )
            print(
                "T =",
                T,
                " search_radius evaluated rate :",
                np.mean(search_res) / h.nb,
            )
            search_res = list(
                pool.imap(
                    search_distortion,
                    ((q, clusters, distortion, T) for q in h.string_q),
                )
            )
            print(
                "T =",
                T,
                " search_distortion evaluated rate :",
                np.mean(search_res) / h.nb,
            )
