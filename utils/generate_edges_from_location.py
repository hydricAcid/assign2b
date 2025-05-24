import pandas as pd
from geopy.distance import geodesic

nodes_path = "data/doc/nodes_averaged.txt"
output_path = "data/doc/input_graph_edges.txt"

DIST_THRESHOLD_KM = 4.0  # km

nodes_df = pd.read_csv(nodes_path)

nodes_df["Road_List"] = nodes_df["Roads"].apply(lambda x: str(x).split("|"))

edges = []

for i in range(len(nodes_df)):
    id1 = nodes_df.iloc[i]["SCATS_ID"]
    coord1 = (nodes_df.iloc[i]["Avg_LAT"], nodes_df.iloc[i]["Avg_LON"])
    roads1 = nodes_df.iloc[i]["Road_List"]

    for j in range(i + 1, len(nodes_df)):
        id2 = nodes_df.iloc[j]["SCATS_ID"]
        coord2 = (nodes_df.iloc[j]["Avg_LAT"], nodes_df.iloc[j]["Avg_LON"])
        roads2 = nodes_df.iloc[j]["Road_List"]

        distance = geodesic(coord1, coord2).km
        if distance <= DIST_THRESHOLD_KM:
            if any(road in roads2 for road in roads1):
                edges.append((id1, id2))
                edges.append((id2, id1))

with open(output_path, "w") as f:
    for src, dst in edges:
        f.write(f"{src} {dst}\n")

print(f"✅ Generated {len(edges)} edges in '{output_path}'")
