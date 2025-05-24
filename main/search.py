from algorithms.a_star import AStar
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.GBFS import GBFS
from algorithms.custom1 import Custom1
from algorithms.custom2 import Custom2
from algorithms.topk import yen_k_shortest_paths


class SearchAlgorithmExecutor:
    def __init__(
        self, algorithm_name, graph, timestamp=None, start=None, goal=None, k=1
    ):
        self.graph = graph
        self.timestamp = timestamp
        self.algorithm_name = algorithm_name.lower()
        self.start = start
        self.goal = goal
        self.k = k

        algorithms = {
            "a_star": AStar,
            "bfs": BFS,
            "dfs": DFS,
            "gbfs": GBFS,
            "custom1": Custom1,
            "custom2": Custom2,
        }

        if self.algorithm_name not in algorithms and self.algorithm_name != "topk":
            raise ValueError(f"Thuật toán không hợp lệ: {self.algorithm_name}")

        self.search_algorithm_class = algorithms.get(self.algorithm_name)

    def search(self, start, goal):
        algorithm = self.search_algorithm_class(self.graph, timestamp=self.timestamp)
        return algorithm.search(start, goal)

    def search_topk(self, start, goal, k):
        return yen_k_shortest_paths(
            self.graph, start, goal, k, self.search_algorithm_class
        )

    def execute(self):
        if self.k > 1:
            return yen_k_shortest_paths(
                self.graph,
                self.start,
                self.goal,
                self.k,
                self.search_algorithm_class,
                self.timestamp,
            )
        else:
            algorithm = self.search_algorithm_class(
                self.graph, timestamp=self.timestamp
            )
            return algorithm.search(self.start, self.goal)
