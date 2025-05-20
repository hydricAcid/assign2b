import pandas as pd
from geopy.distance import geodesic

nodes = pd.read_csv(
    "data/nodes_averaged.txt", sep=" ", header=None, names=["SCATS_ID", "LAT", "LON"]
)

edges = []
threshold = 2  # km

for i in range(len(nodes)):
    id1, lat1, lon1 = nodes.iloc[i]
    for j in range(i + 1, len(nodes)):
        id2, lat2, lon2 = nodes.iloc[j]
        dist = geodesic((lat1, lon1), (lat2, lon2)).km
        if dist <= threshold:
            edges.append((str(id1), str(id2)))

with open("data/input_graph_edges.txt", "w") as f:
    for id1, id2 in edges:
        f.write(f"{id1} {id2}\n")

print(f"✅ Create {len(edges)} edges in file input_graph_edges.txt")
