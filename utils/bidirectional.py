def make_edges_bidirectional(input_file, output_file):
    seen = set()
    with open(input_file, "r") as f:
        lines = f.readlines()

    bidirectional_edges = []
    for line in lines:
        if not line.strip():
            continue
        u, v = line.strip().split()
        edge1 = (u, v)
        edge2 = (v, u)
        if edge1 not in seen:
            bidirectional_edges.append(f"{u} {v}")
            seen.add(edge1)
        if edge2 not in seen:
            bidirectional_edges.append(f"{v} {u}")
            seen.add(edge2)

    with open(output_file, "w") as f:
        for edge in bidirectional_edges:
            f.write(edge + "\n")

    print(
        f"✅ Create bidirectional file at: {output_file} ({len(bidirectional_edges)} cạnh)"
    )


make_edges_bidirectional(
    "data/input_graph_edges.txt", "data/input_graph_edges_bidirectional.txt"
)
