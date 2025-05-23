from graph import Graph
from search import SearchAlgorithmExecutor


def main():
    print("=== Traffic Path Finder ===")
    algo = input("Enter algorithm (a_star, bfs, dfs, gbfs, custom1, custom2): ").strip()
    start_node = input("Enter SCATS ID origin: ").strip()
    goal_node = input("Enter SCATS ID destination: ").strip()

    graph = Graph("data/doc/weighted_edges.txt")

    if not graph.has_node(start_node):
        print(f"❌ Start node {start_node} not found.")
        return
    if not graph.has_node(goal_node):
        print(f"❌ Goal node {goal_node} not found.")
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
