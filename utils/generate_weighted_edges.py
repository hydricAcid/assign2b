import pandas as pd


def load_travel_times(csv_path):
    df = pd.read_csv(csv_path)
    travel_times = df.groupby("SCATS_ID")["TravelTime"].mean().to_dict()
    return travel_times


def assign_weights_to_edges(edge_file, travel_times, output_file, default_time=2.0):
    with open(edge_file, "r") as f:
        lines = f.readlines()

    weighted_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        src, dst = parts

        t1 = travel_times.get(int(src))
        t2 = travel_times.get(int(dst))

        if t1 and t2:
            weight = round((t1 + t2) / 2, 2)
        elif t1:
            weight = round(t1, 2)
        elif t2:
            weight = round(t2, 2)
        else:
            weight = default_time

        weighted_lines.append(f"{src} {dst} {weight}")

    with open(output_file, "w") as f:
        f.write("\n".join(weighted_lines))

    print(f"✅ Ghi file xong: {output_file}")


if __name__ == "__main__":
    travel_time_dict = load_travel_times("evaluation_results/CNN_predictions.csv")
    assign_weights_to_edges(
        edge_file="input_graph_edges.txt",
        travel_times=travel_time_dict,
        output_file="weighted_edges.txt",
    )
