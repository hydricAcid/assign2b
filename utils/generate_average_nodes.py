import pandas as pd
from collections import defaultdict

input_file = "data/input_graph_nodes.txt"
output_file = "data/nodes_averaged.txt"

nodes = defaultdict(lambda: {"lat": [], "lon": []})

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            scats_id = parts[0]
            lat = float(parts[1])
            lon = float(parts[2])
            nodes[scats_id]["lat"].append(lat)
            nodes[scats_id]["lon"].append(lon)

with open(output_file, "w", encoding="utf-8") as f:
    for scats_id, coords in nodes.items():
        avg_lat = sum(coords["lat"]) / len(coords["lat"])
        avg_lon = sum(coords["lon"]) / len(coords["lon"])
        f.write(f"{scats_id} {avg_lat:.6f} {avg_lon:.6f}\n")

print(f"✅ Create file {output_file} with {len(nodes)} SCATS nodes.")
