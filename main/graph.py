class Graph:
    def __init__(self, edge_file="doc/weighted_edges.txt"):
        self.edges = {}
        self.nodes = set()
        self.load_graph(edge_file)

    def load_graph(self, file_path):
        try:
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        src, dest, weight = parts
                        weight = float(weight)
                        self.add_edge(src, dest, weight)
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
        except Exception as e:
            print(f"⚠️ Error loading graph: {e}")

    def add_edge(self, u, v, w):
        if u not in self.edges:
            self.edges[u] = []
        self.edges[u].append((v, w))
        self.nodes.add(u)
        self.nodes.add(v)

    def get_neighbors(self, node):
        return self.edges.get(node, [])

    def has_node(self, node):
        return node in self.nodes
