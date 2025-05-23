from algorithms.a_star import AStar
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.GBFS import GBFS
from algorithms.custom1 import Custom1
from algorithms.custom2 import Custom2
from algorithms.topk import yen_k_shortest_paths


class SearchAlgorithmExecutor:
    def __init__(self, algorithm_name, graph):
        self.graph = graph
        self.algorithm_name = algorithm_name.lower()

        algorithms = {
            "a_star": AStar,
            "bfs": BFS,
            "dfs": DFS,
            "gbfs": GBFS,
            "custom1": Custom1,
            "custom2": Custom2,
        }

        if self.algorithm_name not in algorithms:
            raise ValueError(f"Thuật toán không hợp lệ: {self.algorithm_name}")

        self.search_algorithm_class = algorithms[self.algorithm_name]

    def search(self, start, goal):
        algorithm = self.search_algorithm_class(self.graph)
        return algorithm.search(start, goal)

    def search_topk(self, start, goal, k):
        return yen_k_shortest_paths(
            self.graph, start, goal, k, self.search_algorithm_class
        )
