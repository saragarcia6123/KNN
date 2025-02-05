from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
from heapq import heappush, heappop

from KNN.BallTree.BallNode import BallNode
from KNN.metrics import Metrics

class BallTree:

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

        if prev != self.root:
            prev.add_child(current_node)

        # Pick a random point
        pivot_index = self.random_point(indices)

        # Find the furthest point from the pivot
        distances_from_pivot = self.distances(self.data[pivot_index], self.data[indices])
        point_a_index = indices[np.argmax(distances_from_pivot)]
        point_a = self.data[point_a_index]

        # Find the furthest point from A
        distances_from_a = self.distances(self.data[indices], self.data[point_a_index])
        point_b_index = indices[np.argmax(distances_from_a)]
        point_b = self.data[point_b_index]

        # Join point A and point B to form a line vector
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

    def distances(self, x, y) -> np.ndarray:
        return Metrics.distance_functions[self.metric](x, y)

    def _draw_ball(self, ax: plt.Axes, node: BallNode):

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
        fig: plt.Figure
        ax: plt.Axes

        # Plot the data points from self.data (optional background data)
        ax.scatter(self.data[:, 0], self.data[:, 1], c='gray', alpha=0.5)

        # Begin recursively drawing the balls starting from the root
        self._draw_ball(ax, self.root)

        # Set equal scaling and labels
        ax.set_aspect('equal', 'box')
        ax.set_title('Ball Tree Visualization')
        plt.show()

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

    def _query_point(self, x: np.ndarray, node: BallNode, k: int, min_heap: List[Tuple]):

        # Base case: If the node contains no children
        if len(node.children) == 0:

            # Add its points to minheap
            for p in node.point_indices:
                distance = self.distances([self.data[p]], x)
                heappush(min_heap, (-distance, p))

                if len(min_heap) > k:
                    heappop(min_heap)

            return

        # Find the closest child centroid
        child_distances = self.distances([child.centroid for child in node.children], x)
        closest_child = node.children[np.argmin(child_distances)]
        self._query_point(x, closest_child, k, min_heap)

        # Check if the point on radius coming from the center to the current point
        # is closer than the furthest current element in min_heap

        current_max_dist = -min_heap[0][0]

        for child in node.children:
            if child != closest_child:

                # Join the centroid of the child and the query point to form a line
                line = x - child.centroid

                # Find the unit vector of the line
                line = line / np.linalg.norm(line)

                # Find the point on the line that is at distance child.radius from child.centroid
                point_on_line = child.centroid + (line * child.radius)

                # Find the distance between the point on the line and the query point
                dist = self.distances([point_on_line], x)

                if current_max_dist > dist:
                    self._query_point(x, child, k, min_heap)
