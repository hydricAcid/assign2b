import sys
import time as t
import tracemalloc as tr
from graph import Graph
from algorithms.a_star import AStar
from algorithms.custom1 import Custom1
from algorithms.custom2 import Custom2
from algorithms.dfs import DepthFirstSearch
from algorithms.bfs import BreadthFirstSearch
from algorithms.GBFS import GreedyBestFirstSearch
from algorithms.topk import TopKShortestPaths


def main():
    lines = []
    if len(sys.argv) < 3:
        print("Usage: python search.py <filename> <method> [goal_preference]")
        return

    filename = sys.argv[1]
    method = sys.argv[2].upper()
    goal_preference = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Parse input and create graph
    graph = Graph()
    graph.parse_input(filename)

    if goal_preference is not None and goal_preference in graph.destinations:
        graph.destinations = [goal_preference]

    # Select algorithm based on method parameter
    if method == "AS":
        algorithm = AStar(graph)
    elif method == "CUS2":
        algorithm = Custom2(graph)
    elif method == "DFS":
        algorithm = DepthFirstSearch(graph)
    elif method == "BFS":
        algorithm = BreadthFirstSearch(graph)
    elif method == "CUS1":
        algorithm = Custom1(graph)
    elif method == "GBFS":
        algorithm = GreedyBestFirstSearch(graph)
    elif method == "TOPK":
        k = goal_preference if goal_preference else 5
        algorithm = TopKShortestPaths(graph, k)
        top_k_paths = algorithm.search()
        for i, (path, cost) in enumerate(top_k_paths, 1):
            lines.append(f"Path {i}: {path} - Travel Time: {cost:.2f} sec")
        return lines
    else:
        lines.append(f"Method {method} not supported yet.")
        return lines

    # Run algorithm and get results
    path, num_nodes = algorithm.search()

    # Print results

    lines.append(f"{filename} {method}")
    if path:
        goal = path[-1]
        lines.append(f"{goal} {num_nodes}")
        lines.append(f"[{','.join(map(str, path))}]")
    else:
        lines.append("No path found")
    return lines


if __name__ == "__main__":
    startTime = t.time()  # runtime and memory measuring
    tr.start()

    lines = main()

    endTime = t.time()
    memUse = tr.get_traced_memory()
    tr.stop()

    for c in lines:
        print(c)

    lines.append(f"Memory Usage: {memUse}")
    lines.append(f"Execution Time: {(endTime-startTime) * 10**3} ms")

    f = open("log.txt", "a")  # writes verbose information to a log file
    f.write("----------NEW ENTRY----------\n")
    for c in lines:
        f.write(str(c))
        f.write("\n")
    f.close()
