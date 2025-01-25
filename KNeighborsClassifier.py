from typing import Optional, Self
import numpy as np

from BallTree import BallTree


class KNeighborsClassifier:

    def __init__(self,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        algorithm: str = "ball_tree",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = 'minkowski',
        metric_params: Optional[dict] = None,
        n_jobs: Optional[int] = None
    ):
        self.n_outputs = None
        self.n_features = None
        self.n_samples = None
        self._tree = None
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray):

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same n_samples")

        if len(y.shape) > 2:
            raise ValueError("y cannot contain more than 2 dimensions")

        if self.metric == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError()
        else:
            self.n_samples = X.shape[0]
            self.n_features = X.shape[1]

        if len(y.shape) == 2:
            self.n_outputs = y.shape[1]

        if self.algorithm == 'ball_tree':
            self._tree = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
        elif self.algorithm == 'kd_tree':
            raise NotImplementedError()
        elif self.algorithm == 'brute':
            raise NotImplementedError()
        else:
            raise ValueError("Invalid algorithm")

    def kneighbors(self, X: np.ndarray):

        if self.algorithm == 'ball_tree':
            return self._tree.query_points(X, k=self.n_neighbors)

        elif self.algorithm == 'kd_tree':
            raise NotImplementedError()
        elif self.algorithm == 'brute':
            raise NotImplementedError()
        else:
            raise ValueError("Invalid algorithm")

    def visualize(self):
        if self.algorithm == 'ball_tree':
            self._tree.visualize()