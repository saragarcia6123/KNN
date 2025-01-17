from typing import List
import numpy as np
from matplotlib import pyplot as plt
import metrics
from BallNode import BallNode
from heapq import heappush, heappop

class BallTree:

    valid_metrics = ['euclidean', 'minkowski', 'manhattan', 'chebyshev']
    distance_functions = {
        'euclidean': metrics.euclidean_distances,
        'minkowski': metrics.minkowski_distances,
        'manhattan': metrics.manhattan_distances,
        'chebyshev': metrics.chebyshev_distances,
    }

    def __init__(self,
        X: np.ndarray,
        leaf_size: int = 40,
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
        self.generate_tree()

    def generate_tree(self):
        indices = [i for i in range(self.n_samples)]
        self.root = self.create_ball_node(indices)
        self.split(indices, self.root)

    def create_ball_node(self, indices: List) -> BallNode:

        points = self.data[indices]

        # Calculate the center point by finding the middle point between the two furthest points
        centroid = np.mean(points, axis=0)

        # Find the radius as the maximum distance from any point to the center
        radius = max(np.linalg.norm(point - centroid) for point in points)

        return BallNode(centroid, radius, indices)

    def split(self, indices: List, prev: BallNode):

        if len(indices) <= self.leaf_size:
            return

        # Add the current node to the previous node
        current_node = self.create_ball_node(indices)
        prev.add_child(current_node)

        # Pick a random point
        pivot_index = self.random_point(indices)

        # Find the furthest point from pivot
        distances_from_pivot = self.distances(pivot_index, indices)
        point_a_index = np.argmax(distances_from_pivot)

        # Find the furthest point from A
        distances_from_a = self.distances(indices, point_a_index)
        point_b_index = np.argmax(distances_from_a)

        point_a = self.data[point_a_index]
        point_b = self.data[point_b_index]

        # Join point A and point B to form a line
        line = point_b - point_a

        # Project the points onto the line
        projections = np.dot(self.data[indices] - point_a, line) / (np.linalg.norm(line)**2)

        # Split along the median
        median = np.median(projections)

        # Form 2 spheres on either side of the median
        s1_indices = [indices[i] for i in range(len(indices)) if projections[i] <= median]
        s2_indices = [indices[i] for i in range(len(indices)) if projections[i] > median]

        # Recursively split the points
        self.split(s1_indices, current_node)
        self.split(s2_indices, current_node)

    def random_point(self, x):
        r_index = np.random.randint(0, len(x))
        return x[r_index]

    def distances(self, x, y):
        return self.distance_functions[self.metric](self.data[x], self.data[y])

    def _draw_ball(self, ax, node: BallNode):

        if node.is_leaf():
            return

        # Get centroid and radius of the current node (ball)
        x, y = node.centroid
        radius = node.radius

        if node == self.root:
            radius = 0

        # Draw the ball as a circle
        circle = plt.Circle((x, y), radius, color='blue', alpha=0.3)  # Semi-transparent circle
        ax.add_artist(circle)

        # Recursively draw the balls of the children
        for child in node.children:
            self._draw_ball(ax, child)

    def visualize(self):
        fig, ax = plt.subplots()

        # Plot the data points from self.data (optional background data)
        ax.scatter(self.data[:, 0], self.data[:, 1], c='gray', alpha=0.5)

        # Begin recursively drawing the balls starting from the root
        self._draw_ball(ax, self.root)

        # Set equal scaling and labels
        ax.set_aspect('equal', 'box')
        ax.set_title('Ball Tree Visualization')
        plt.show()

    def query(self, X: np.ndarray, k: int) -> List:
        result_indices = []
        for x in X:
            result = self._query(x, k)
            result_indices.append(result)

        results = [self.data[indices] for indices in result_indices]

        return results

    def _query(self, point: np.ndarray, k: int) -> List:

        # If the node has less than k points, return all the points in the node
        if len(self.root.get_points()) <= k:
            return [(np.linalg.norm(point - p), p) for p in self.root.get_points()]

        # Store the nearest k neighbors and their distances
        min_heap = []

        self._recursive_search(point, self.root, k, min_heap)

        # Extract the k nearest neighbors from the heap
        k_nearest = [heappop(min_heap) for _ in range(k)]
        k_nearest = [point for _, point in k_nearest]

        return k_nearest

    def _recursive_search(self, point: np.ndarray, node: BallNode, k: int, min_heap: List):

        if node.is_leaf():
            for p in node.get_points():
                # Calculate the distance between the query point and the current point
                dist = np.linalg.norm(point - p)

                # Add the distance and point to the min-heap
                heappush(min_heap, (-dist, p))

                if len(min_heap) > k:
                    # If the heap has more than k elements, remove the element with the largest distance
                    heappop(min_heap)

        children = node.get_children()

        if not children:
            return

        # Find the child whose centroid is closest to the query point
        closest_child = min(children, key=lambda child: np.linalg.norm(point - child.centroid))

        # Search the closest child first
        self._recursive_search(point, closest_child, k, min_heap)

        # After exploring the closest child, check the other child nodes
        for child in children:
            if child != closest_child:
                # Check the current largest distance in the heap
                largest_dist = -min_heap[0][0]

                # Check the distance of the centroid of the child node to the query point
                dist_to_centroid = np.linalg.norm(point - child.centroid)

                # If the distance to the centroid is less than the current largest distance in the heap
                # or if the heap has less than k elements, search the child node
                if dist_to_centroid < largest_dist or len(min_heap) < k:
                    self._recursive_search(point, child, k, min_heap)
