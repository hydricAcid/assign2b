import pandas as pd


def generate_nodes_from_scats(xlsx_path, output_path):
    # Đọc dữ liệu từ dòng 3 (index=2), không dùng dòng đó làm header
    df = pd.read_excel(xlsx_path, sheet_name="Data", header=2)

    # Tự đặt lại tên cột
    df.columns = [
        "SCATS Number",
        "Location",
        "CD_MELWAY",
        "NB_LATITUDE",
        "NB_LONGITUDE",
        "HF VicRoads Internal",
        "VR Internal Stat",
        "VR Internal Loc",
        "NB_TYPE_SURVEY",
    ] + [
        f"Data{i}" for i in range(1, len(df.columns) - 9 + 1)
    ]  # phần còn lại là dữ liệu lưu lượng

    print("✅ Cột sau khi gán lại:")
    print(df.columns[:10].tolist())  # kiểm tra 10 cột đầu tiên

    # Lấy ra duy nhất các SCATS node
    scats_col = "SCATS Number"
    lat_col = "NB_LATITUDE"
    lon_col = "NB_LONGITUDE"

    unique_nodes = df[[scats_col, lat_col, lon_col]].drop_duplicates()

    node_lines = ["Nodes:"]
    for _, row in unique_nodes.iterrows():
        try:
            scats = str(int(row[scats_col])).zfill(4)
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            node_lines.append(f"{scats}: ({lat:.6f}, {lon:.6f})")
        except Exception as e:
            print(f"Bỏ qua dòng lỗi: {e}")

    with open(output_path, "w") as f:
        f.write("\n".join(node_lines))
    print(f"✅ Ghi file node xong: {output_path}")


if __name__ == "__main__":
    generate_nodes_from_scats(
        xlsx_path="data/raw/Scats_data_october_2006.xlsx",
        output_path="input_graph_nodes.txt",
    )
