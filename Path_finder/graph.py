import os

class Graph:
    def __init__(self):
        self.nodes = {}  # node_id -> (x, y)
        self.edges = {}  # from_node -> {to_node: cost}
        self.origin = None
        self.destinations = []

    def parse_input(self, filename):
        """Parse input file and construct the graph"""
        
        if os.path.exists(f"{filename}"): # check if file exists
            filepath = f"{filename}"
        else:
            print("File not found, aborting")
            exit()

        
        with open(filepath, "r") as file:
            lines = file.readlines()
            mode = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("Nodes:"):
                    mode = "nodes"
                elif line.startswith("Edges:"):
                    mode = "edges"
                elif line.startswith("Origin:"):
                    mode = "origin"
                elif line.startswith("Destinations:"):
                    mode = "destinations"
                else:
                    if mode == "nodes":
                        parts = line.split(":")
                        node = int(parts[0].strip())
                        coords = parts[1].strip()[1:-1].split(",")
                        x, y = map(int, coords)
                        self.nodes[node] = (x, y)
                        self.edges[node] = {}  # Initialize empty adjacency list

                    elif mode == "edges":
                        parts = line.split(":")
                        edge = parts[0].strip()[1:-1].split(",")
                        from_node, to_node = map(int, edge)
                        cost = int(parts[1].strip())
                        self.edges[from_node][to_node] = cost

                    elif mode == "origin":
                        self.origin = int(line)

                    elif mode == "destinations":
                        dests = line.split(";")
                        self.destinations = [int(d.strip()) for d in dests]

    def get_neighbors(self, node):
        """Get neighbors of a node sorted by node_id"""
        if node in self.edges:
            neighbors = list(self.edges[node].items())
            neighbors.sort(key=lambda x: x[0])  # Sort by node_id
            return neighbors
        return []

    def get_cost(self, from_node, to_node):
        """Get cost of an edge"""
        if from_node in self.edges and to_node in self.edges[from_node]:
            return self.edges[from_node][to_node]
        return float("inf")

    def get_coordinates(self, node):
        """Get coordinates of a node"""
        return self.nodes.get(node)

    def is_destination(self, node):
        """Check if node is a destination"""
        return node in self.destinations

    def get_total_nodes(self):
        # gets total number of nodes in the graph
        return len(self.nodes)
