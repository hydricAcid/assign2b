import math

from algorithms.search_algorithm import SearchAlgorithm


class DepthFirstSearch(SearchAlgorithm):

    def __init__(self, graph):
        super().__init__(graph)


    def search(self, depth_limit=None):

        start_node = self.graph.origin

        visited = set()
        stack = [(start_node, [start_node], 0)]  # (node, path, current_depth)

        while stack:
            current_node, path, current_depth = stack.pop()
            self.node_count += 1
            if current_node in visited:
                continue
            visited.add(current_node)
            

            if self.is_goal(current_node):
                return path, self.node_count

            # depth is for iddfs, has no role in normal dfs
            if depth_limit is None or current_depth < depth_limit:
                neighbors = self.get_neighbors(current_node)
                for neighbor, _ in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor], current_depth + 1))

        return None, self.node_count
