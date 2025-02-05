from typing import List, Self

class BallNode:

    def __init__(self, centroid, radius, point_indices):
        self.centroid = centroid
        self.radius = radius
        self.children: List[BallNode] = []
        self.point_indices = point_indices

    def add_child(self, child: Self.__class__):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0