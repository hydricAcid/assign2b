import math

from .search_algorithm import SearchAlgorithm
from collections import deque


class BFS(SearchAlgorithm):
    def __init__(self, graph):
        self.graph = graph

    def search(self, start, goal):
        from collections import deque

        queue = deque([(start, [start])])
        visited = set()

        while queue:
            current, path = queue.popleft()

            if current == goal:
                return path, len(path)

            if current not in visited:
                visited.add(current)
                for neighbor, _ in self.graph.get_neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))

        return None, float("inf")
