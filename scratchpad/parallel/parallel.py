# one way
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        res = p.map(f, [1, 2, 3])

res
multiprocessing.cpu_count()

# or another - threading
