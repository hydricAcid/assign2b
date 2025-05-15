import heapq
import math
from algorithms.search_algorithm import SearchAlgorithm


class Custom2(SearchAlgorithm):
    def __init__(self, graph):
        self.graph = graph

    def search(self, start, goal):
        # Uniform Cost Search
        import heapq

        queue = [(0, start, [start])]
        visited = set()

        while queue:
            cost, current, path = heapq.heappop(queue)

            if current == goal:
                return path, cost

            if current not in visited:
                visited.add(current)
                for neighbor, weight in self.graph.get_neighbors(current):
                    if neighbor not in visited:
                        heapq.heappush(
                            queue, (cost + weight, neighbor, path + [neighbor])
                        )

        return None, float("inf")
