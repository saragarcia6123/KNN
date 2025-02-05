from typing import List, Self

class KDNode:

    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.idx = idx
        self.children: List[KDNode] = []

    def add_child(self, child: Self.__class__):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0