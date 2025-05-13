class SearchAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.node_count = 0

    def search(self):
        """Base search method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement search method")

    def is_goal(self, node):
        """Check if node is a goal state"""
        return self.graph.is_destination(node)

    def get_neighbors(self, node):
        """Get neighbors of a node sorted by node_id"""
        return self.graph.get_neighbors(node)
