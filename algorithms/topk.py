import os
import sys
from heapq import heappush, heappop
from copy import deepcopy

sys.path.append(os.path.abspath("main"))
from graph import Graph
from algorithms.a_star import AStar


def yen_k_shortest_paths(graph: Graph, start: str, goal: str, k: int = 3):
    if start == goal:
        return [(0.0, [start])]

    shortest_paths = []
    potential_paths = []

    base_path, base_cost = AStar(graph).search(start, goal)
    if not base_path:
        return []

    shortest_paths.append((base_cost, base_path))

    for i in range(1, k):
        for j in range(len(shortest_paths[-1][1]) - 1):
            spur_node = shortest_paths[-1][1][j]
            root_path = shortest_paths[-1][1][: j + 1]

            temp_graph = deepcopy(graph)

            for cost, path in shortest_paths:
                if path[: j + 1] == root_path and len(path) > j + 1:
                    u, v = path[j], path[j + 1]
                    temp_graph.edges[u] = [
                        (n, w) for (n, w) in temp_graph.edges.get(u, []) if n != v
                    ]

            for node in root_path[:-1]:
                temp_graph.edges.pop(node, None)

            spur_path, spur_cost = AStar(temp_graph).search(spur_node, goal)

            if spur_path and spur_node in spur_path:
                full_path = root_path[:-1] + spur_path
                total_cost = graph.calculate_path_cost(full_path)
                if (total_cost, full_path) not in potential_paths:
                    heappush(potential_paths, (total_cost, full_path))

        if not potential_paths:
            break

        while potential_paths:
            next_path = heappop(potential_paths)
            if next_path not in shortest_paths:
                shortest_paths.append(next_path)
                break

    return shortest_paths
