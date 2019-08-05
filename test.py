from multiprocessing import Pool
import numpy as np


def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        a = p.map(f, range(5))
        print(np.array(a))