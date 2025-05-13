import math

from algorithms.search_algorithm import SearchAlgorithm
from algorithms.dfs import DepthFirstSearch

# IDDFS Implementation (Iterative Deepening Depth First Search)
class Custom1(SearchAlgorithm):

    def __init__(self, graph):
        super().__init__(graph)

    def search(self):

        start_node = self.graph.origin
        depth = 0
        while True:
            
            dfs = DepthFirstSearch(self.graph)
            path, self.node_count = dfs.search(depth_limit=depth)

            if path:  
                return path, self.node_count

            depth += 1 

            if depth > self.graph.get_total_nodes(): # just in case there is no destination, prevents from running forever
                return None, self.node_count
