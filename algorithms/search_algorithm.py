class SearchAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.node_count = 0

    def search(self):
        raise NotImplementedError("Subclasses must implement search method")

    def is_goal(self, node):
        return self.graph.is_destination(node)

    def get_neighbors(self, node):
        return self.graph.get_neighbors(
            node, self.timestamp if hasattr(self, "timestamp") else None
        )
