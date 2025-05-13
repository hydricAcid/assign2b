import heapq
from algorithms.search_algorithm import SearchAlgorithm

class GreedyBestFirstSearch(SearchAlgorithm):
    def __init__(self, graph):
        super().__init__(graph)
        # Calculate heuristic for all destinations
        self.heuristics = {
            node: self._calculate_heuristic(node) 
            for node in self.graph.destinations
        }

    def _calculate_heuristic(self, node):
        """Calculate Manhattan distance heuristic from node to nearest destination"""
        x1, y1 = self.graph.get_coordinates(node)
        min_distance = float('inf')
        
        for dest in self.graph.destinations:
            x2, y2 = self.graph.get_coordinates(dest)
            distance = abs(x1 - x2) + abs(y1 - y2)
            if distance < min_distance:
                min_distance = distance
                
        return min_distance
    
    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def search(self):
        frontier = []
        heapq.heappush(frontier, (self._calculate_heuristic(self.graph.origin), self.graph.origin))
        came_from = {}
        visited = set()

        while frontier:
            _, current = heapq.heappop(frontier)
            self.node_count += 1

            if self.graph.is_destination(current):
                return self._reconstruct_path(came_from, current), self.node_count

            if current in visited:
                continue
            visited.add(current)

            for neighbor, _ in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    heapq.heappush(frontier, (self._calculate_heuristic(neighbor), neighbor))

        return None, self.node_count
