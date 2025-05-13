import heapq
import math
from algorithms.search_algorithm import SearchAlgorithm


class AStar(SearchAlgorithm):
    def __init__(self, graph):
        super().__init__(graph)

    def heuristic(self, node, destinations):
        """Manhattan distance heuristic for A*"""
        x1, y1 = self.graph.get_coordinates(node)
        min_distance = float("inf")

        for goal in destinations:
            x2, y2 = self.graph.get_coordinates(goal)
            distance = abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def search(self):
        """A* search algorithm implementation"""
        open_set = []
        # (f, node_id, current, path, g) - sort by f first, then by node_id
        start_node = self.graph.origin
        heapq.heappush(open_set, (0, start_node, start_node, [start_node], 0))

        visited = set()
        self.node_count = 0

        while open_set:
            _, node_id, current, path, g = heapq.heappop(open_set)
            self.node_count += 1

            if self.is_goal(current):
                return path, self.node_count

            if current in visited:
                continue

            visited.add(current)

            for neighbor, cost in self.get_neighbors(current):
                if neighbor not in visited:
                    new_g = g + cost
                    new_f = new_g + self.heuristic(neighbor, self.graph.destinations)
                    heapq.heappush(
                        open_set, (new_f, neighbor, neighbor, path + [neighbor], new_g)
                    )

        return None, self.node_count
