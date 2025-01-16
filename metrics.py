import numpy as np

def euclidean_distances(x: np.ndarray, y: np.ndarray):
    return np.sqrt(np.sum((x - y)**2, axis=1))

def manhattan_distances(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x - y), axis=1)

def chebyshev_distances(x: np.ndarray, y: np.ndarray):
    return np.max(np.abs(x - y), axis=1)

def minkowski_distances(x: np.ndarray, y: np.ndarray, p: float = 2):
    return np.power(np.sum(np.power(np.abs(x - y), p), axis=1), 1/p)