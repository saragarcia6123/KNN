from typing import List, Self

class BallNode:

    def __init__(self, centroid, radius, points):
        self.centroid = centroid
        self.radius = radius
        self.children: List[BallNode] = []
        self.points = points

    def add_child(self, child: Self.__class__):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0