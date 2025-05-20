import pandas as pd


def generate_weighted_edges(
    edges_path="data/input_graph_edges_bidirectional.txt",
    cnn_predictions_path="evaluation_results/CNN_predictions.csv",
    output_path="data/weighted_edges.txt",
):
    edges = []
    with open(edges_path, "r") as f:
        for line in f:
            src, dst = line.strip().split()
            src = str(int(float(src)))
            dst = str(int(float(dst)))
            edges.append((src, dst))

    df_pred = pd.read_csv(cnn_predictions_path)
    if "SCATS_ID" not in df_pred.columns or "TravelTime" not in df_pred.columns:
        raise ValueError("CSV file must contain 'SCATS_ID' and 'TravelTime' columns")

    df_pred["SCATS_ID"] = df_pred["SCATS_ID"].astype(str)

    avg_tt = df_pred.groupby("SCATS_ID")["TravelTime"].mean().to_dict()

    weighted_edges = []
    for src, dst in edges:
        if src in avg_tt and dst in avg_tt:
            weight = round((avg_tt[src] + avg_tt[dst]) / 2, 2)
            weighted_edges.append((src, dst, weight))

    with open(output_path, "w") as f:
        for src, dst, weight in weighted_edges:
            f.write(f"{src} {dst} {weight}\n")

    print(f"✅ Create {len(weighted_edges)} edges with weighted: {output_path}")


if __name__ == "__main__":
    generate_weighted_edges()
