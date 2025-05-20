import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time as t
import tracemalloc as tr
from graph import Graph
from algorithms.a_star import AStar
from algorithms.custom1 import Custom1
from algorithms.custom2 import Custom2
from algorithms.dfs import DFS
from algorithms.bfs import BFS
from algorithms.GBFS import GBFS
from algorithms.topk import yen_k_shortest_paths


class SearchAlgorithmExecutor:
    def __init__(self, graph, algorithm_name):
        self.graph = graph
        self.algorithm_name = algorithm_name.lower()
        self.algorithm = self._get_algorithm()

    def _get_algorithm(self):
        if self.algorithm_name == "a_star":
            return AStar(self.graph)
        elif self.algorithm_name == "bfs":
            return BFS(self.graph)
        elif self.algorithm_name == "dfs":
            return DFS(self.graph)
        elif self.algorithm_name == "gbfs":
            return GBFS(self.graph)
        elif self.algorithm_name == "custom1":
            return Custom1(self.graph)
        elif self.algorithm_name == "custom2":
            return Custom2(self.graph)
        elif self.algorithm_name == "topk":
            return None  # handled separately
        else:
            raise ValueError(f"Algorithm not supported: {self.algorithm_name}")

    def search(self, start, goal):
        if self.algorithm_name == "topk":
            raise ValueError("Using search_topk for top-k algorithm")
        return self.algorithm.search(start, goal)

    def search_topk(self, start, goal, k):
        paths = yen_k_shortest_paths(self.graph, start, goal, k)
        if not paths:
            return []
        return [(path_info[1], path_info[0]) for path_info in paths]
