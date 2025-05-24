import pandas as pd
from collections import defaultdict


class Graph:
    def __init__(self):
        self.nodes = {}  # SCATS_ID -> (lat, lon)
        self.edges_by_time = defaultdict(
            lambda: defaultdict(list)
        )  # timestamp -> src -> list of (dst, weight)

    def load_nodes(self, filepath):
        self.nodes.clear()
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            try:
                node_id = str(int(row["SCATS_ID"]))
                lat = float(row["Avg_LAT"])
                lon = float(row["Avg_LON"])
                self.nodes[node_id] = (lat, lon)
            except (ValueError, KeyError):
                continue

    def load_weighted_edges_with_timestamp(self, filepath):
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["source"] = df["source"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()

        for _, row in df.iterrows():
            src = row["source"]
            dst = row["target"]
            timestamp = row["timestamp"].strftime("%Y-%m-%d %H:%M:00")
            weight = float(row["travel_time_min"])
            self.edges_by_time[timestamp][src].append((dst, weight))

    def get_neighbors(self, node, timestamp=None):
        if timestamp and timestamp in self.edges_by_time:
            neighbors = self.edges_by_time[timestamp].get(node, [])
            # print(f"👀 Neighbors of {node} at {timestamp}: {neighbors}")
            return neighbors
        print(f"⚠️ No edges found for timestamp: {timestamp}")
        return []

    def heuristic(self, node1, node2):
        if node1 not in self.nodes or node2 not in self.nodes:
            return float("inf")
        lat1, lon1 = self.nodes[node1]
        lat2, lon2 = self.nodes[node2]
        return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

    def calculate_path_cost(self, path, timestamp=None):
        total = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for neighbor, weight in self.get_neighbors(u, timestamp):
                if neighbor == v:
                    total += weight
                    break
        return total

    def has_node(self, node_id):
        return node_id in self.nodes

    def highlight_path(self, canvas, path, color="red"):
        scaled_positions = self.get_scaled_positions(
            canvas.winfo_width(), canvas.winfo_height()
        )
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in scaled_positions and v in scaled_positions:
                x1, y1 = scaled_positions[u]
                x2, y2 = scaled_positions[v]
                canvas.create_line(x1, y1, x2, y2, fill=color, width=5)

    def get_scaled_positions(self, canvas_width, canvas_height):
        if not self.nodes:
            return {}

        lats = [coord[0] for coord in self.nodes.values()]
        lons = [coord[1] for coord in self.nodes.values()]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        margin = 40

        scale_x = (canvas_width - 2 * margin) / lon_range
        scale_y = (canvas_height - 2 * margin) / lat_range

        scaled = {}
        for node, (lat, lon) in self.nodes.items():
            x = margin + (lon - min_lon) * scale_x
            y = canvas_height - (margin + (lat - min_lat) * scale_y)
            scaled[node] = (x, y)

        return scaled
