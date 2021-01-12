import random

from tqdm import tqdm
from functools import partial
from Levenshtein import distance
from multiprocessing import Pool, cpu_count


def calculate_medoid(items):
    return min(items, key=partial(total_distance, items))


def total_distance(items, medoid):
    return sum(distance(medoid, o) for o in items)

def associate_(args_):
    o, medoids = args_
    medoid = min(medoids, key=lambda m: distance(m, o))    
    return medoid

def associate(pool, medoids, items):
    assigned = list(pool.imap(associate_, [(x, medoids) for x in items]))
    clusters = {medoid: [] for medoid in medoids}
    for x, ci in zip(items, assigned):
        clusters[ci].append(x)
    return clusters

def cost(solution):
    return sum(
        total_distance(items, medoid) for medoid, items in solution.items()
    )

def clustered(items, n_clusters, iterations):
    """
    Implements K-Medoids clustering
    """
    with Pool(cpu_count()) as pool:
        # start off clustering around random medoids
        medoids = random.sample(items, n_clusters)

        clusters = associate(pool, medoids, items)


        for i in tqdm(range(iterations)):
            previous =  clusters
            # update medoids
            # medoids = [calculate_medoid(cluster) for cluster in clusters.values()]
            medoids = list(pool.imap(calculate_medoid, clusters.values()))
            # recompute clusters with new medoids
            clusters = associate(pool, medoids, items)
            # has the solution converged?
            if cost(clusters) == cost(previous):
                break
    
    print(cost(clusters) / len(items))
    return clusters


def k_medoids_embedding(args, h):
    clusters = clustered(h.string_b, 1024, 20)
