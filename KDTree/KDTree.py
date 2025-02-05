from heapq import heappop, heappush
from typing import List, Tuple
import numpy as np

from KNN.KDTree.KDNode import KDNode
from KNN.metrics import Metrics

class KDTree:

    def __init__(self,
         X: np.ndarray,
         leaf_size: int = 2,
         metric: str = 'minkowski'
    ):
        if leaf_size < 2:
            raise ValueError("Minimum leaf size is 2")

        self.data = X
        self.points = None
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.leaf_size = leaf_size
        self.metric = metric
        self.root = None
        self._generate_tree()

    def _generate_tree(self):

        # Select dimension of highest variance for sorting
        variances = np.var(self.data, axis=0)
        dim = np.argsort(variances)[-1]

        indices = np.argsort(self.data[:, dim])
        root_index = int(indices[len(self.data) // 2])

        self._split(indices, root_index, self.root, dim)

    def _create_node(self, idx: int) -> KDNode:
        x, y = self.data[idx]
        return KDNode(x, y, idx)

    def _split(self, indices, prev_index: int, parent_node: KDNode, dimension) -> None:

        current_node = self._create_node(prev_index)

        if parent_node != self.root:
            parent_node.add_child(current_node)

        if len(indices) <= self.leaf_size:
            return

        sorted_data = self.data[indices]
        median_idx = len(sorted_data) // 2

        left_data = sorted_data[:median_idx]
        right_data = sorted_data[median_idx+1:]  # Points > median

        next_dim = (dimension + 1) % len(self.data[0])

        self._split(left_data, median_idx, parent_node, next_dim)
        self._split(right_data, median_idx, parent_node, next_dim)

    def _distances(self, x, y) -> np.ndarray:
        return Metrics.distance_functions[self.metric](x, y)

    def query_points(self, X: np.ndarray, k: int) -> List:
        results = []

        # Find the k-nearest neighbors for each query point
        for x in X:
            min_heap = []
            self._query_point(x, self.root, k, min_heap)
            k_nearest = [heappop(min_heap) for _ in range(k)]
            indexes = [val for _, val in k_nearest]

            # Get the corresponding data points for the indexes in the min_heap
            points = [self.data[i] for i in indexes]
            results.append(points)

        return results

    def _query_point(self, x: np.ndarray, node: KDNode, k: int, min_heap: List[Tuple]):

        # Base case: If the node contains no children
        if len(node.children) == 0:

            # TODO

            return