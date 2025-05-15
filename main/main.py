from graph import Graph
from search import SearchAlgorithmExecutor


def main():
    print("=== Traffic Path Finder ===")
    algo = input("Nhập thuật toán (a_star, bfs, dfs, gbfs, custom1, custom2): ").strip()
    start_node = input("Nhập SCATS ID bắt đầu: ").strip()
    goal_node = input("Nhập SCATS ID kết thúc: ").strip()

    graph = Graph("data/weighted_edges.txt")

    if not graph.has_node(start_node):
        print(f"❌ Start node {start_node} không tồn tại.")
        return
    if not graph.has_node(goal_node):
        print(f"❌ Goal node {goal_node} không tồn tại.")
        return

    executor = SearchAlgorithmExecutor(graph, algo)
    path, cost = executor.search(start_node, goal_node)

    if not path:
        print(f"❌ No path found from {start_node} to {goal_node}")
    else:
        print(f"✅ Path found using {algo.upper()}:")
        print(" → ".join(path))
        print(f"🕒 Total travel time: {cost:.2f} minutes")


if __name__ == "__main__":
    main()
