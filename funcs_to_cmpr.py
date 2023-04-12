from proj_consts import *

import ray
import numpy as np

def get_random_sqr_tensor(dim = DIM_OF_SQUARE, amnt = AMNT_OF_NDARRAYS):
    return np.random.rand(amnt, dim, dim)

@ray.remote
def get_ray_random_sqr_tensor(dim = DIM_OF_SQUARE, amnt = AMNT_OF_NDARRAYS):
    return np.random.rand(amnt, dim, dim)

def get_random_sqr_matrx(dim = DIM_OF_SQUARE):
    return np.random.rand(dim, dim)

@ray.remote
def get_ray_random_sqr_matrx(dim = DIM_OF_SQUARE):
    return np.random.rand(dim, dim)

def add_sqr_matrx(mtrx1: np.ndarray, mtrx2: np.ndarray):
    assert mtrx1.shape == mtrx2.shape, f'Matrices does not have equal shape'
    return np.add(mtrx1, mtrx2)

@ray.remote
def add_sqr_ray_matrx(mtrx1: np.ndarray, mtrx2: np.ndarray):
    assert mtrx1.shape == mtrx2.shape, f'Matrices does not have equal shape'
    return np.add(mtrx1, mtrx2)

def dot_prod_sqr_matrx(mtrx1: np.ndarray, mtrx2: np.ndarray):
    assert mtrx1.shape == mtrx2.shape, f'Matrices does not have equal shape'
    return np.dot(mtrx1, mtrx2)

@ray.remote
def dot_prod_ray_sqr_matrx(mtrx1: np.ndarray, mtrx2: np.ndarray):
    assert mtrx1.shape == mtrx2.shape, f'Matrices does not have equal shape'
    return np.dot(mtrx1, mtrx2)

def sum_matrices(np_tensor: np.ndarray):
    assert len(np_tensor) >= 2, f'Tensor does not have enough elements to perform operation'
    for idx in range(len(np_tensor)):
        if idx == 0 :
            temp_np_tensor = np_tensor[idx].copy()
        else:
            temp_np_tensor += np_tensor[idx]
    return temp_np_tensor

@ray.remote
def sum_ray_matrices(np_tensor: np.ndarray):
    assert len(np_tensor) >= 2, f'Tensor does not have enough elements to perform operation'
    for idx in range(len(np_tensor)):
        if idx == 0 :
            temp_np_tensor = np_tensor[idx].copy()
        else:
            temp_np_tensor += np_tensor[idx]
    return temp_np_tensor.copy()

def dot_product_matrices(np_tensor: np.ndarray):
    assert len(np_tensor) >= 2, f'Tensor does not have enough elements to perform operation'
    for idx in range(len(np_tensor)):
        if idx == 0 :
            temp_np_tensor = np_tensor[idx]
        else:
            temp_np_tensor = np.dot(temp_np_tensor, np_tensor[idx])
    return temp_np_tensor

@ray.remote
def dot_product_ray_matrices(np_tensor: np.ndarray):
    assert len(np_tensor) >= 2, f'Tensor does not have enough elements to perform operation'
    for idx in range(len(np_tensor)):
        if idx == 0 :
            temp_np_tensor = np_tensor[idx]
        else:
            temp_np_tensor = np.dot(temp_np_tensor, np_tensor[idx])
    return temp_np_tensor
