import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.append(os.path.abspath("main"))
from graph import Graph
from search import SearchAlgorithmExecutor
from algorithms.topk import yen_k_shortest_paths


class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Path Finder")
        self.canvas_width = 1000
        self.canvas_height = 700

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        self.control_frame = tk.Frame(main_frame)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.canvas = tk.Canvas(
            main_frame, width=self.canvas_width, height=self.canvas_height, bg="white"
        )
        self.canvas.pack(side="right", fill="both", expand=True)

        self.graph = Graph()
        self.graph.load_nodes("data/nodes_averaged.txt")
        self.graph.load_weighted_edges("data/weighted_edges.txt")

        ttk.Label(self.control_frame, text="Start Node:").pack(anchor="w")
        self.start_entry = ttk.Entry(self.control_frame)
        self.start_entry.pack(fill="x")

        ttk.Label(self.control_frame, text="Goal Node:").pack(anchor="w", pady=(10, 0))
        self.goal_entry = ttk.Entry(self.control_frame)
        self.goal_entry.pack(fill="x")

        ttk.Label(self.control_frame, text="Algorithm:").pack(anchor="w", pady=(10, 0))
        self.algorithm_combo = ttk.Combobox(
            self.control_frame,
            values=["a_star", "bfs", "dfs", "gbfs", "custom1", "custom2", "topk"],
        )
        self.algorithm_combo.set("a_star")
        self.algorithm_combo.pack(fill="x")
        self.algorithm_combo.bind("<<ComboboxSelected>>", self.on_algorithm_change)

        self.k_label = ttk.Label(self.control_frame, text="K Paths:")
        self.k_entry = ttk.Entry(self.control_frame)

        self.find_btn = ttk.Button(
            self.control_frame, text="Find Path", command=self.find_path
        )
        self.find_btn.pack(pady=20)

        self.draw_graph()

    def on_algorithm_change(self, event=None):
        algo = self.algorithm_combo.get().lower()
        if algo == "topk":
            self.k_label.pack(anchor="w")
            self.k_entry.pack(fill="x")
        else:
            self.k_label.pack_forget()
            self.k_entry.pack_forget()

    def scale_coordinates(self, lat, lon):
        lats = [coord[0] for coord in self.graph.nodes.values()]
        lons = [coord[1] for coord in self.graph.nodes.values()]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        padding = 50
        scale_x = (self.canvas_width - 2 * padding) / (max_lon - min_lon)
        scale_y = (self.canvas_height - 2 * padding) / (max_lat - min_lat)

        x = padding + (lon - min_lon) * scale_x
        y = self.canvas_height - (padding + (lat - min_lat) * scale_y)
        return x, y

    def draw_graph(self):
        self.canvas.delete("all")
        for node, (lat, lon) in self.graph.nodes.items():
            x, y = self.scale_coordinates(lat, lon)
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="black")
            self.canvas.create_text(
                x + 6, y - 6, text=node, anchor="nw", font=("Arial", 8)
            )

        self.edge_lines = {}
        for from_node, neighbors in self.graph.edges.items():
            for to_node, _ in neighbors:
                if from_node in self.graph.nodes and to_node in self.graph.nodes:
                    x1, y1 = self.scale_coordinates(*self.graph.nodes[from_node])
                    x2, y2 = self.scale_coordinates(*self.graph.nodes[to_node])
                    line = self.canvas.create_line(x1, y1, x2, y2, fill="gray")
                    self.edge_lines[(from_node, to_node)] = line
        print(f"✅ Draw {len(self.edge_lines)} edges.")

    def highlight_path(self, path, color="blue"):
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            line = self.edge_lines.get((u, v)) or self.edge_lines.get((v, u))
            if line:
                self.canvas.itemconfig(line, fill=color, width=3)

    def find_path(self):
        start = self.start_entry.get().strip()
        goal = self.goal_entry.get().strip()
        algo = self.algorithm_combo.get().lower()

        if not start or not goal:
            messagebox.showerror("Error", "Enter Start and Goal Node.")
            return

        executor = SearchAlgorithmExecutor(self.graph, algo)

        if algo == "topk":
            try:
                k = int(self.k_entry.get().strip())
                paths = executor.search_topk(start, goal, k)
                if not paths:
                    messagebox.showinfo("Result", "❌ No path found.")
                    return
                print("✅ Debug paths:", paths)

                self.draw_graph()
                for i, result in enumerate(paths):
                    if isinstance(result, tuple) and len(result) == 2:
                        path, cost = result
                        color = "blue" if i == 0 else "green"
                        self.highlight_path(path, color)
                messagebox.showinfo("Result", f"✅ Find {len(paths)} path.")
            except ValueError:
                messagebox.showerror("Error", "Please enter an integer for K.")

        else:
            path, cost = executor.search(start, goal)
            if not path:
                messagebox.showinfo("Result", "❌ No path found.")
                return
            self.draw_graph()
            self.highlight_path(path)
            messagebox.showinfo("Result", f"✅ Total time: {cost:.2f} minutes")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()
