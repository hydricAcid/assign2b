import math


class Graph:
    def __init__(self):
        self.nodes = {}  # SCATS_ID -> (lat, lon)
        self.edges = {}  # SCATS_ID -> list of (neighbor_id, weight)

    def load_nodes(self, filepath):
        self.nodes.clear()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    node_id = parts[0]
                    try:
                        lat = float(parts[1])
                        lon = float(parts[2])
                        self.nodes[node_id] = (lat, lon)
                    except ValueError:
                        continue

    def load_weighted_edges(self, filepath):
        self.edges.clear()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    src, dst, weight_str = parts
                    try:
                        weight = float(weight_str)
                        self.edges.setdefault(src, []).append((dst, weight))
                    except ValueError:
                        continue

    def get_neighbors(self, node):
        return self.edges.get(node, [])

    def heuristic(self, node1, node2):
        if node1 not in self.nodes or node2 not in self.nodes:
            return float("inf")
        lat1, lon1 = self.nodes[node1]
        lat2, lon2 = self.nodes[node2]
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    def calculate_path_cost(self, path):
        total = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for neighbor, weight in self.edges.get(u, []):
                if neighbor == v:
                    total += weight
                    break
        return total
