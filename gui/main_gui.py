import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.append(os.path.abspath("main"))
from graph import Graph
from search import SearchAlgorithmExecutor


class TrafficApp:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Route Finder")

        # GUI controls
        tk.Label(master, text="Start SCATS ID:").grid(row=0, column=0, padx=10, pady=5)
        self.start_entry = tk.Entry(master)
        self.start_entry.grid(row=0, column=1)

        tk.Label(master, text="Goal SCATS ID:").grid(row=1, column=0, padx=10, pady=5)
        self.goal_entry = tk.Entry(master)
        self.goal_entry.grid(row=1, column=1)

        tk.Label(master, text="Algorithm:").grid(row=2, column=0, padx=10, pady=5)
        self.algorithm_cb = ttk.Combobox(
            master, values=["A_STAR", "BFS", "DFS", "GBFS", "CUSTOM1", "CUSTOM2"]
        )
        self.algorithm_cb.current(0)
        self.algorithm_cb.grid(row=2, column=1)

        self.find_button = tk.Button(master, text="Find Path", command=self.find_path)
        self.find_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.canvas = tk.Canvas(master, width=800, height=600, bg="white")
        self.canvas.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def find_path(self):
        start = self.start_entry.get().strip()
        goal = self.goal_entry.get().strip()
        algorithm = self.algorithm_cb.get().strip().upper()

        try:
            graph = Graph()
            graph.load_nodes_file("doc/input_graph_nodes.txt")
            graph.load_weighted_edges_file("doc/weighted_edges.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load graph files: {e}")
            return

        executor = SearchAlgorithmExecutor(algorithm)
        executor.set_graph(graph)
        path, cost = executor.search(start, goal)

        if not path:
            messagebox.showwarning("No Path", f"No path found from {start} to {goal}")
            return

        messagebox.showinfo(
            "Route Found",
            f"Path: {' → '.join(path)}\nTotal travel time: {cost:.2f} minutes",
        )
        self.draw_graph(graph, path)

    def draw_graph(self, graph, path=None):
        self.canvas.delete("all")
        r = 5  # node radius
        color_path = "#FF5733"
        color_node = "#4682B4"

        # Draw edges
        count = 0
        for from_node, neighbors in graph.edges.items():
            if from_node not in graph.node_positions:
                continue
            x1, y1 = graph.node_positions[from_node]
            for to_node in neighbors:
                if to_node not in graph.node_positions:
                    continue
                x2, y2 = graph.node_positions[to_node]
                self.canvas.create_line(x1, y1, x2, y2, fill="gray")
                count += 1

        if count == 0:
            print(
                "⚠️ Không có cạnh nào được vẽ. Có thể SCATS ID trong edges không khớp với node positions."
            )
            print("Edges:", list(graph.edges.items())[:5])
            print("Nodes:", list(graph.node_positions.items())[:5])

        # Draw nodes
        for node, (x, y) in graph.node_positions.items():
            fill = color_node
            if path and node in path:
                fill = color_path
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=fill, outline="")
            self.canvas.create_text(x, y - 10, text=node, font=("Arial", 8))

        # Draw path lines
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                if a in graph.node_positions and b in graph.node_positions:
                    x1, y1 = graph.node_positions[a]
                    x2, y2 = graph.node_positions[b]
                    self.canvas.create_line(x1, y1, x2, y2, fill=color_path, width=3)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()
