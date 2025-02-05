import numpy as np

def _euclidean_distances(x: np.ndarray, y: np.ndarray):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))

def _manhattan_distances(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x - y), axis=1)

def _chebyshev_distances(x: np.ndarray, y: np.ndarray):
    return np.max(np.abs(x - y), axis=1)

def _minkowski_distances(x: np.ndarray, y: np.ndarray, p: float = 2):
    return np.power(np.sum(np.power(np.abs(x - y), p), axis=1), 1 / p)

class Metrics:

    valid_metrics = ['euclidean', 'minkowski', 'manhattan', 'chebyshev']

    distance_functions = {
        'euclidean': _euclidean_distances,
        'minkowski': _minkowski_distances,
        'manhattan': _manhattan_distances,
        'chebyshev': _chebyshev_distances,
    }