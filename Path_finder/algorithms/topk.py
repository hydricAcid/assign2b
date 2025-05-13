import networkx as nx
from algorithms.search_algorithm import SearchAlgorithm


class TopKShortestPaths(SearchAlgorithm):
    def __init__(self, graph, k=5):
        super().__init__(graph)
        self.k = k

    def search(self):
        origin = self.graph.origin
        destinations = self.graph.destinations
        edges = self.graph.edges

        G = nx.DiGraph()
        for from_node, neighbors in edges.items():
            for to_node, cost in neighbors.items():
                G.add_edge(from_node, to_node, weight=cost)

        top_k_paths = []
        for dest in destinations:
            try:
                all_paths = nx.shortest_simple_paths(G, origin, dest, weight="weight")
                for i, path in enumerate(all_paths):
                    if i >= self.k:
                        break
                    total_cost = sum(
                        G[path[j]][path[j + 1]]["weight"] for j in range(len(path) - 1)
                    )
                    top_k_paths.append((path, total_cost))
            except nx.NetworkXNoPath:
                continue

        top_k_paths.sort(key=lambda x: x[1])
        return top_k_paths
