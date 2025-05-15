import math

from algorithms.search_algorithm import SearchAlgorithm


class Custom1(SearchAlgorithm):
    def __init__(self, graph):
        self.graph = graph

    def search(self, start, goal):
        # DFS with depth limit (5)
        limit = 5
        return self.dls(start, goal, limit)

    def dls(self, current, goal, limit, path=None, visited=None):
        if visited is None:
            visited = set()
        if path is None:
            path = [current]
        visited.add(current)

        if current == goal:
            return path, len(path)

        if limit <= 0:
            return None, float("inf")

        for neighbor, _ in self.graph.get_neighbors(current):
            if neighbor not in visited:
                result, cost = self.dls(
                    neighbor, goal, limit - 1, path + [neighbor], visited
                )
                if result:
                    return result, cost

        return None, float("inf")
