import pandas as pd
import numpy as np
import math


def solve_speed_from_flow(flow):

    a = -1.4648375
    b = 93.75
    c = -flow

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return 60.0

    sqrt_discriminant = math.sqrt(discriminant)
    v1 = (-b + sqrt_discriminant) / (2 * a)
    v2 = (-b - sqrt_discriminant) / (2 * a)

    for v in [v1, v2]:
        if 0 < v <= 100:
            return v

    return 60.0


def flow_to_travel_time(flow_array, timestamp_array=None, distance_km=1.0):
    travel_times = []

    for i, flow in enumerate(flow_array):
        ts = timestamp_array[i] if timestamp_array is not None else None

        if flow <= 351:
            speed = 60.0
        else:
            speed = solve_speed_from_flow(flow)

        travel_time = (distance_km / speed) * 60

        if ts is not None:
            hour = pd.to_datetime(ts).hour
            if 7 <= hour <= 9 or 16 <= hour <= 18:
                travel_time *= 1.2

        travel_times.append(travel_time)

    return np.array(travel_times)


def flow_to_travel_time_single(flow, timestamp, distance_km=1.0):
    if flow <= 351:
        speed = 60.0
    else:
        speed = solve_speed_from_flow(flow)

    travel_time = (distance_km / speed) * 60
    hour = pd.to_datetime(timestamp).hour
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        travel_time *= 1.2
    return travel_time


def convert_predictions_to_travel_time(input_csv, output_csv, distance_km=1.0):
    df = pd.read_csv(input_csv)
    df["Predicted_Travel_Time_min"] = df.apply(
        lambda row: flow_to_travel_time_single(
            row["Predicted"], row["Timestamp"], distance_km
        ),
        axis=1,
    )
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved updated file with travel time to: {output_csv}")


if __name__ == "__main__":
    convert_predictions_to_travel_time(
        input_csv="evaluation_results/CNN_predictions.csv",
        output_csv="evaluation_results/CNN_predictions_with_tt.csv",
        distance_km=1.0,
    )
