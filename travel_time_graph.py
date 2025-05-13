import pandas as pd
from geopy.distance import geodesic
from math import sqrt
from sklearn.neighbors import NearestNeighbors


# === Flow to speed conversion function (from assignment spec) ===
def flow_to_speed(flow):
    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return 0

    sqrt_disc = sqrt(discriminant)
    s1 = (-b + sqrt_disc) / (2 * a)
    s2 = (-b - sqrt_disc) / (2 * a)

    speed = max(s1, s2)
    return min(speed, 60) if speed > 0 else 0


# === Travel time estimation ===
def estimate_travel_time(distance_km, flow):
    speed = flow_to_speed(flow)
    if speed <= 0:
        return float("inf")
    base_time_hr = distance_km / speed
    return base_time_hr * 3600 + 30  # seconds + 30s delay


# === Load SCATS site data ===
location_path = "data/Traffic_Count_Locations_with_LONG_LAT.csv"
df = pd.read_csv(location_path)

# === Filter for Boroondara area (rough bounding box) ===
df_boro = df[
    (df["Y"] >= -37.85)
    & (df["Y"] <= -37.75)
    & (df["X"] >= 144.95)
    & (df["X"] <= 145.10)
].copy()

# === Prepare coordinates and IDs ===
df_boro = df_boro.rename(columns={"FID": "scats_id", "X": "lon", "Y": "lat"})
coords = df_boro[["lat", "lon"]].values

# === Use Nearest Neighbors to build edge list ===
nbrs = NearestNeighbors(n_neighbors=3).fit(coords)
distances, indices = nbrs.kneighbors(coords)

edges = []
for i in range(len(df_boro)):
    site = df_boro.iloc[i]
    site_id = site["scats_id"]
    lat1, lon1 = site["lat"], site["lon"]

    for j in indices[i][1:]:
        neighbor = df_boro.iloc[j]
        neighbor_id = neighbor["scats_id"]
        lat2, lon2 = neighbor["lat"], neighbor["lon"]

        dist_km = geodesic((lat1, lon1), (lat2, lon2)).km
        flow = 800  # Placeholder
        speed = flow_to_speed(flow)
        travel_time = estimate_travel_time(dist_km, flow)

        edges.append(
            {
                "from": int(site_id),
                "to": int(neighbor_id),
                "distance_km": round(dist_km, 3),
                "predicted_flow": flow,
                "estimated_speed_kmh": round(speed, 2),
                "travel_time_sec": round(travel_time, 2),
            }
        )

# === Save output ===
edges_df = pd.DataFrame(edges)
edges_df.to_csv("boro_travel_time_graph.csv", index=False)
print("✅ Travel time graph saved to 'boro_travel_time_graph.csv'")
