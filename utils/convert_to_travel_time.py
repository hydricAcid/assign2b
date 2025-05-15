import pandas as pd
import os


def flow_to_travel_time(flow):
    if flow <= 200:
        return 1.0
    elif flow <= 600:
        return 1.5
    elif flow <= 1000:
        return 2.0
    elif flow <= 1500:
        return 3.0
    else:
        return 4.0


def convert_predictions_to_travel_time(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df["Predicted_Travel_Time_min_per_km"] = df["Predicted"].apply(flow_to_travel_time)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Travel time saved to {output_csv}")


if __name__ == "__main__":
    convert_predictions_to_travel_time(
        input_csv="evaluation_results/CNN_predictions.csv",
        output_csv="evaluation_results/CNN_travel_time.csv",
    )

convert_flow_to_travel_time = flow_to_travel_time
