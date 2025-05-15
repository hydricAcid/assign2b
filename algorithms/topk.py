import heapq
import copy


class TopKShortestPaths:
    def __init__(self, graph, k):
        self.graph = graph
        self.k = k

    def dijkstra(self, source, target, banned_edges):
        visited = set()
        dist = {node: float("inf") for node in self.graph.nodes}
        prev = {node: None for node in self.graph.nodes}
        dist[source] = 0
        heap = [(0, source)]

        while heap:
            cost, current = heapq.heappop(heap)
            if current == target:
                break
            if current in visited:
                continue
            visited.add(current)

            for neighbor, weight in self.graph.edges.get(current, {}).items():
                if (current, neighbor) in banned_edges:
                    continue
                alt = cost + weight
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prev[neighbor] = current
                    heapq.heappush(heap, (alt, neighbor))

        # Reconstruct path
        path = []
        node = target
        while node is not None:
            path.insert(0, node)
            node = prev[node]
        if path and path[0] == source:
            return path, dist[target]
        else:
            return None, float("inf")

    def search(self):
        source = self.graph.origin
        target = self.graph.destinations[0]
        paths = []

        # First shortest path
        path, cost = self.dijkstra(source, target, set())
        if not path:
            return []
        paths.append((path, cost))
        candidates = []

        for k in range(1, self.k):
            for i in range(len(paths[0][0]) - 1):
                spur_node = paths[0][0][i]
                root_path = paths[0][0][: i + 1]

                banned_edges = set()
                for p, _ in paths:
                    if p[: i + 1] == root_path and i + 1 < len(p):
                        banned_edges.add((p[i], p[i + 1]))

                spur_path, spur_cost = self.dijkstra(spur_node, target, banned_edges)
                if spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_cost = 0
                    for i in range(len(total_path) - 1):
                        u, v = total_path[i], total_path[i + 1]
                        total_cost += self.graph.get_cost(u, v)
                    candidate = (total_path, total_cost)
                    if candidate not in candidates and candidate not in paths:
                        candidates.append(candidate)

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1])
            paths.append(candidates.pop(0))

        return paths
