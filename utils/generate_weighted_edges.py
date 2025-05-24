import pandas as pd

prediction_file = "evaluation_results/CNN_predictions_with_tt.csv"
edge_file = "data/doc/input_graph_edges.txt"
output_file = "data/doc/weighted_edges_per_timestamp.csv"

df_pred = pd.read_csv(prediction_file)

df_pred["Timestamp"] = pd.to_datetime(df_pred["Timestamp"])

with open(edge_file, "r") as f:
    edges = [line.strip().split() for line in f.readlines()]
    edges = [(int(src), int(dst)) for src, dst in edges]

grouped = df_pred.groupby(["SCATS_ID", "Timestamp"])["Predicted_Travel_Time_min"].mean()

results = []

all_timestamps = sorted(df_pred["Timestamp"].unique())

for timestamp in all_timestamps:
    for src, dst in edges:
        try:
            src_tt = grouped[(src, timestamp)]
            dst_tt = grouped[(dst, timestamp)]
            avg_tt = round((src_tt + dst_tt) / 2, 2)
            results.append(
                {
                    "source": src,
                    "target": dst,
                    "timestamp": timestamp,
                    "travel_time_min": avg_tt,
                }
            )
        except KeyError:
            continue

df_result = pd.DataFrame(results)
df_result.to_csv(output_file, index=False)

print(f"✅ Generated weighted edge file with timestamps: {output_file}")
