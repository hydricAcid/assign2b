import math

from algorithms.search_algorithm import SearchAlgorithm
from collections import deque

class BreadthFirstSearch(SearchAlgorithm):

    def __init__(self, graph):
        super().__init__(graph)


    def search(self):
        start_node = self.graph.origin

        visited = set()
        queue = deque([(start_node, [start_node])])

        while queue:
            current_node, path = queue.popleft()

            if current_node in visited:
                continue
            
            visited.add(current_node)
            self.node_count += 1

            if self.is_goal(current_node):
                return path, self.node_count

            neighbors = self.get_neighbors(current_node)

            for neighbor, _ in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None, self.node_count
