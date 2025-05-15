import heapq
import math
from algorithms.search_algorithm import SearchAlgorithm


class AStar(SearchAlgorithm):
    def __init__(self, graph):
        self.graph = graph

    def search(self, start, goal):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current), g_score[current]

            for neighbor, weight in self.graph.get_neighbors(current):
                tentative_g = g_score[current] + weight
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g  # No heuristic used
                    heapq.heappush(open_set, (f, neighbor))

        return None, float("inf")

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path
