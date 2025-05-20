import pandas as pd

excel_path = "data/processed/modified_scats_data_oct_2006.xlsx"
sheet_name = "Data"
output_path = "data/input_graph_nodes.txt"

df = pd.read_excel(excel_path, sheet_name=sheet_name)

df.columns = [str(col).strip() for col in df.columns]

required_columns = ["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]
if not all(col in df.columns for col in required_columns):
    print("❌ Can not find columns:", df.columns)
    exit()

df_clean = df.dropna(subset=required_columns).drop_duplicates(
    subset=["SCATS Number", "Location", "NB_LATITUDE", "NB_LONGITUDE"]
)

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_clean.iterrows():
        scats_id = str(row["SCATS Number"]).strip()
        lat = row["NB_LATITUDE"]
        lon = row["NB_LONGITUDE"]
        location = (
            str(row["Location"]).strip().replace(" ", "_")
        )  # Thay space bằng underscore
        f.write(f"{scats_id} {lat} {lon} {location}\n")

print(f"✅ Create file {output_path} with {len(df_clean)} node.")
