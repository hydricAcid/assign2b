import heapq
import math
from algorithms.search_algorithm import SearchAlgorithm


class Custom2(SearchAlgorithm):
    def __init__(self, graph):
        super().__init__(graph)

    def heuristic_steps(self, node, destinations):
        """Heuristic for CUS2: Estimate minimum steps to nearest goal"""
        x1, y1 = self.graph.get_coordinates(node)
        min_distance = float("inf")

        for goal in destinations:
            x2, y2 = self.graph.get_coordinates(goal)
            # Use Euclidean distance divided by a constant to not overestimate
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / 3
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def search(self):
        """Custom Search 2: Informed search focusing on minimizing steps"""
        open_set = []
        # (priority, node_id, steps_so_far, current, path)
        start_node = self.graph.origin
        initial_heuristic = self.heuristic_steps(start_node, self.graph.destinations)
        heapq.heappush(
            open_set, (initial_heuristic, start_node, 0, start_node, [start_node])
        )

        visited = {}  # node -> best steps so far
        self.node_count = 0

        while open_set:
            _, node_id, steps, current, path = heapq.heappop(open_set)
            self.node_count += 1

            if self.is_goal(current):
                return path, self.node_count

            # If node was visited with fewer or equal steps, skip
            if current in visited and visited[current] <= steps:
                continue

            visited[current] = steps

            for neighbor, _ in self.get_neighbors(current):  # Ignoring edge cost
                new_steps = steps + 1
                if neighbor not in visited or new_steps < visited[neighbor]:
                    h = self.heuristic_steps(neighbor, self.graph.destinations)
                    priority = new_steps + h  # f = g + h where g is steps
                    heapq.heappush(
                        open_set,
                        (priority, neighbor, new_steps, neighbor, path + [neighbor]),
                    )

        return None, self.node_count
