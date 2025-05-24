import math

from .search_algorithm import SearchAlgorithm


class DFS(SearchAlgorithm):
    def __init__(self, graph, timestamp=None):
        self.graph = graph
        self.timestamp = timestamp

    def search(self, start, goal):
        stack = [(start, [start])]
        visited = set()

        while stack:
            current, path = stack.pop()

            if current == goal:
                return path, len(path)

            if current not in visited:
                visited.add(current)
                for neighbor, _ in self.graph.get_neighbors(
                    current, timestamp=self.timestamp
                ):

                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))

        return None, float("inf")
