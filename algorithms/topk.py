import os
import sys
from heapq import heappush, heappop
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main.graph import Graph


def yen_k_shortest_paths(
    graph: Graph,
    start: str,
    goal: str,
    k: int = 3,
    search_algorithm_class=None,
    timestamp=None,
):
    if start == goal:
        return [(0.0, [start])]

    if search_algorithm_class is None:
        raise ValueError("A search algorithm class must be provided.")

    shortest_paths = []
    potential_paths = []

    base_algorithm = search_algorithm_class(graph, timestamp=timestamp)
    base_path, base_cost = base_algorithm.search(start, goal)
    if not base_path:
        return []

    shortest_paths.append((base_cost, base_path))
    visited_nodes = set(base_path)

    for i in range(1, k):
        for j in range(len(shortest_paths[-1][1]) - 1):
            spur_node = shortest_paths[-1][1][j]
            root_path = shortest_paths[-1][1][: j + 1]

            temp_graph = deepcopy(graph)

            for cost, path in shortest_paths:
                if path[: j + 1] == root_path and len(path) > j + 1:
                    u, v = path[j], path[j + 1]
                    temp_neighbors = temp_graph.edges_by_time.get(timestamp, {}).get(
                        u, []
                    )
                    temp_graph.edges_by_time[timestamp][u] = [
                        (n, w) for (n, w) in temp_neighbors if n != v
                    ]

            for node in root_path[:-1]:
                if node in temp_graph.edges_by_time.get(timestamp, {}):
                    temp_graph.edges_by_time[timestamp].pop(node, None)

            spur_algorithm = search_algorithm_class(temp_graph, timestamp=timestamp)
            spur_path, spur_cost = spur_algorithm.search(spur_node, goal)

            if spur_path and spur_node in spur_path:
                full_path = root_path[:-1] + spur_path

                new_nodes = set(full_path) - visited_nodes
                if not new_nodes:
                    continue

                total_cost = graph.calculate_path_cost(full_path, timestamp=timestamp)
                full_path_tuple = tuple(full_path)
                if all(tuple(p[1]) != full_path_tuple for p in potential_paths):
                    heappush(potential_paths, (total_cost, full_path))

        if not potential_paths:
            break

        while potential_paths:
            next_path = heappop(potential_paths)
            if all(tuple(p[1]) != tuple(next_path[1]) for p in shortest_paths):
                shortest_paths.append(next_path)
                visited_nodes.update(next_path[1])
                break

    return shortest_paths
