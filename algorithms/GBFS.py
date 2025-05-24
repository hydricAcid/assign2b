import heapq
from .search_algorithm import SearchAlgorithm


class GBFS(SearchAlgorithm):
    def __init__(self, graph, timestamp=None):
        self.graph = graph
        self.timestamp = timestamp

    def search(self, start, goal):
        queue = [(0, start, [start])]
        visited = set()

        while queue:
            _, current, path = heapq.heappop(queue)

            if current == goal:
                return path, len(path)

            if current not in visited:
                visited.add(current)
                for neighbor, _ in self.graph.get_neighbors(
                    current, timestamp=self.timestamp
                ):
                    if neighbor not in visited:
                        h = self.graph.heuristic(neighbor, goal)
                        heapq.heappush(queue, (h, neighbor, path + [neighbor]))

        return None, float("inf")
