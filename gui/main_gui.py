import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from main.search import SearchAlgorithmExecutor
from main.graph import Graph

from algorithms.topk import yen_k_shortest_paths


class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Route Finder with K-Paths")

        self.canvas_width = 1100
        self.canvas_height = 800

        self.graph = Graph()
        self.graph.load_nodes("data/doc/nodes_averaged.txt")
        self.graph.load_weighted_edges("data/doc/weighted_edges.txt")

        self.create_widgets()
        self.draw_graph()

    def create_widgets(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(control_frame, text="Start Node:").pack()
        self.start_entry = tk.Entry(control_frame)
        self.start_entry.pack()

        tk.Label(control_frame, text="Goal Node:").pack()
        self.goal_entry = tk.Entry(control_frame)
        self.goal_entry.pack()

        tk.Label(control_frame, text="K (Top Paths):").pack()
        self.k_entry = tk.Entry(control_frame)
        self.k_entry.insert(0, "1")
        self.k_entry.pack()

        tk.Label(control_frame, text="Algorithm:").pack()
        self.algorithm_combo = ttk.Combobox(
            control_frame, values=["a_star", "bfs", "dfs", "gbfs", "custom1", "custom2"]
        )
        self.algorithm_combo.current(0)
        self.algorithm_combo.pack()

        tk.Button(control_frame, text="Find Path", command=self.find_path).pack(pady=10)

        self.canvas = tk.Canvas(
            self.root, width=self.canvas_width, height=self.canvas_height, bg="white"
        )
        self.canvas.pack(side=tk.RIGHT)

    def get_scaled_positions(self):
        lats = [coord[0] for coord in self.graph.nodes.values()]
        lons = [coord[1] for coord in self.graph.nodes.values()]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        margin = 40

        scale_x = (self.canvas_width - 2 * margin) / lon_range if lon_range != 0 else 1
        scale_y = (self.canvas_height - 2 * margin) / lat_range if lat_range != 0 else 1

        scaled = {}
        for node, (lat, lon) in self.graph.nodes.items():
            x = margin + (lon - min_lon) * scale_x
            y = self.canvas_height - (margin + (lat - min_lat) * scale_y)
            scaled[node] = (x, y)
        return scaled

    def draw_graph(self):
        self.canvas.delete("all")
        positions = self.get_scaled_positions()

        # Draw edges
        for node, neighbors in self.graph.edges.items():
            for neighbor, _ in neighbors:
                x1, y1 = positions[node]
                x2, y2 = positions[neighbor]
                self.canvas.create_line(x1, y1, x2, y2, fill="gray")

        # Draw nodes
        for node, (x, y) in positions.items():
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")
            self.canvas.create_text(
                x + 5,
                y - 5,
                text=str(node),
                anchor=tk.NW,
                font=("Arial", 9, "bold"),
                fill="darkblue",
            )

        print("✅ Draw", len(self.graph.edges), "edges.")

    def highlight_path(self, path, color="blue"):
        positions = self.get_scaled_positions()
        for i in range(len(path) - 1):
            x1, y1 = positions[path[i]]
            x2, y2 = positions[path[i + 1]]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

    def find_path(self):
        start = self.start_entry.get().strip()
        goal = self.goal_entry.get().strip()
        algo = self.algorithm_combo.get().lower()
        k_text = self.k_entry.get().strip()

        if not start or not goal or not algo:
            messagebox.showerror("Error", "Enter all fields.")
            return

        if not self.graph.has_node(start) or not self.graph.has_node(goal):
            messagebox.showerror("Error", "Start or goal node not found.")
            return

        try:
            k = int(k_text)
            if k <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "K must be a positive integer.")
            return

        executor = SearchAlgorithmExecutor(algo, self.graph)
        try:
            paths = executor.search_topk(start, goal, k)
            if not paths:
                messagebox.showinfo("Result", "❌ No path found.")
                return

            self.draw_graph()
            colors = ["blue", "green", "red", "purple", "orange", "cyan"]
            result_msg = f"✅ Found {len(paths)} path(s) from {start} to {goal}:\n\n"
            for i, (cost, path) in enumerate(paths):
                color = colors[i % len(colors)]
                self.highlight_path(path, color)
                result_msg += f"🔹 Path {i+1} (Time: {cost:.2f} minutes):\n"
                result_msg += " → ".join(path) + "\n\n"

            messagebox.showinfo("Result", result_msg)

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()
