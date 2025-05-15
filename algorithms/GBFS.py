import heapq
from .search_algorithm import SearchAlgorithm


class GBFS(SearchAlgorithm):
    def __init__(self, graph):
        self.graph = graph

    def search(self, start, goal):
        import heapq

        queue = [(0, start, [start])]
        visited = set()

        while queue:
            _, current, path = heapq.heappop(queue)

            if current == goal:
                return path, len(path)

            if current not in visited:
                visited.add(current)
                for neighbor, weight in self.graph.get_neighbors(current):
                    heapq.heappush(queue, (weight, neighbor, path + [neighbor]))

        return None, float("inf")
