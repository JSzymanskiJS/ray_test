#%%
from pprint import pprint
from datetime import datetime

import ray
import numpy as np
import pandas as pd
#%%
ray.init()
#%%
AMNT_OF_CORES_TO_UTIL = 8
AMNT_OF_NDARRAYS = 2 * AMNT_OF_CORES_TO_UTIL
DIM_OF_SQUARE = 14000
DIM_OF_NPARRAY = (DIM_OF_SQUARE, DIM_OF_SQUARE)
ITERATIONS = [ 17]
#%%
def get_random_sqr_matrx(dim = DIM_OF_SQUARE):
    return np.random.rand(dim, dim)

def add_sqr_matrx():
    mtrx1 = get_random_sqr_matrx()
    mtrx2 = get_random_sqr_matrx()
    assert mtrx1.shape == mtrx2.shape, f'Matrices does not have equal shape'
    return np.add(mtrx1, mtrx2)

@ray.remote
def add_sqr_ray_matrx():
    mtrx1 = get_random_sqr_matrx()
    mtrx2 = get_random_sqr_matrx()
    assert mtrx1.shape == mtrx2.shape, f'Matrices does not have equal shape'
    return np.add(mtrx1, mtrx2)
#%%
def no_ray(iterations: int = 3):
    for i in range(iterations):
        add_sqr_matrx()

def with_ray(iterations: int = 3):
    futures = [add_sqr_ray_matrx.remote() for _ in range(iterations)]
    ray.get(futures)

def with_ray_return(iterations: int = 3):
    futures = [add_sqr_ray_matrx.remote() for _ in range(iterations)]
    return ray.get(futures)
#%%
def time_func(func, *args, **kwargs):
    start = datetime.now()
    func(*args, **kwargs)
    return datetime.now() - start
#%%
funcs = [no_ray, with_ray, with_ray_return]
comparison = [[time_func(func, i) for i in ITERATIONS] for func in funcs]
#%%
funcs_names = [func.__name__ for func in funcs]
df = pd.DataFrame(
    data=np.array(comparison).T,
    columns=funcs_names
)
#%%
print(df)
#%%
