import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from main.search import SearchAlgorithmExecutor
from main.graph import Graph

from algorithms.topk import yen_k_shortest_paths

import datetime


class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Flow Simulation")

        self.graph = Graph()
        self.graph.load_nodes("data/doc/nodes_averaged.txt")
        self.graph.load_weighted_edges_with_timestamp(
            "data/doc/weighted_edges_per_timestamp.csv"
        )

        self.canvas_width = 1000
        self.canvas_height = 800
        self.canvas = tk.Canvas(
            root, width=self.canvas_width, height=self.canvas_height, bg="white"
        )
        self.canvas.grid(row=0, column=1, rowspan=20)

        self.controls_frame = ttk.Frame(root)
        self.controls_frame.grid(row=0, column=0, sticky="nw", padx=10, pady=10)

        self.setup_controls()
        self.draw_graph()

    def setup_controls(self):
        ttk.Label(self.controls_frame, text="Start Node:").grid(
            row=0, column=0, sticky="w"
        )
        self.start_entry = ttk.Entry(self.controls_frame)
        self.start_entry.grid(row=1, column=0, sticky="ew")

        ttk.Label(self.controls_frame, text="Goal Node:").grid(
            row=2, column=0, sticky="w"
        )
        self.goal_entry = ttk.Entry(self.controls_frame)
        self.goal_entry.grid(row=3, column=0, sticky="ew")

        ttk.Label(self.controls_frame, text="Timestamp (YYYY-MM-DD HH:MM):").grid(
            row=4, column=0, sticky="w"
        )
        self.timestamp_entry = ttk.Entry(self.controls_frame)
        self.timestamp_entry.grid(row=5, column=0, sticky="ew")
        self.timestamp_entry.insert(0, "2006-10-01 08:00")

        ttk.Label(self.controls_frame, text="Algorithm:").grid(
            row=6, column=0, sticky="w"
        )
        self.algo_combo = ttk.Combobox(
            self.controls_frame,
            values=["a_star", "bfs", "dfs", "gbfs", "custom1", "custom2"],
        )
        self.algo_combo.grid(row=7, column=0, sticky="ew")
        self.algo_combo.set("a_star")

        ttk.Label(self.controls_frame, text="K (for Top-K):").grid(
            row=8, column=0, sticky="w"
        )
        self.k_entry = ttk.Entry(self.controls_frame)
        self.k_entry.grid(row=9, column=0, sticky="ew")
        self.k_entry.insert(0, "3")

        self.find_button = ttk.Button(
            self.controls_frame, text="Find Path", command=self.find_path
        )
        self.find_button.grid(row=10, column=0, pady=10)

    def draw_graph(self):
        self.canvas.delete("all")
        scaled_positions = self.graph.get_scaled_positions(
            self.canvas_width, self.canvas_height
        )

        seen_edges = set()
        for timestamp_data in self.graph.edges_by_time.values():
            for src, neighbors in timestamp_data.items():
                for dst, _ in neighbors:
                    edge = tuple(sorted((src, dst)))
                    if edge not in seen_edges:
                        seen_edges.add(edge)
                        if src in scaled_positions and dst in scaled_positions:
                            x1, y1 = scaled_positions[src]
                            x2, y2 = scaled_positions[dst]
                            self.canvas.create_line(x1, y1, x2, y2, fill="lightgray")

        for node, (x, y) in scaled_positions.items():
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="blue")
            self.canvas.create_text(
                x, y - 10, text=str(node), fill="black", font=("Arial", 10, "bold")
            )

    def find_path(self):
        start = self.start_entry.get().strip()
        goal = self.goal_entry.get().strip()
        timestamp_str = self.timestamp_entry.get().strip()
        algo = self.algo_combo.get()
        k = int(self.k_entry.get()) if self.k_entry.get().isdigit() else 3

        try:
            timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
        except ValueError:
            messagebox.showerror(
                "Error", "Invalid timestamp format. Use YYYY-MM-DD HH:MM"
            )
            return

        formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:00")

        if start not in self.graph.nodes or goal not in self.graph.nodes:
            messagebox.showerror("Error", "Start or Goal node not found in graph")
            return

        executor = SearchAlgorithmExecutor(
            algorithm_name=algo,
            graph=self.graph,
            timestamp=formatted_timestamp,
            start=start,
            goal=goal,
            k=k,
        )

        result = executor.execute()

        color_list = ["red", "green", "orange", "purple", "brown", "blue", "cyan"]
        scaled_positions = self.graph.get_scaled_positions(
            self.canvas_width, self.canvas_height
        )

        self.draw_graph()

        if result is None or result == []:
            messagebox.showwarning("No Path", "No path found between the nodes")
            return

        if isinstance(result, list):

            result_window = tk.Toplevel(self.root)
            result_window.title(f"Top-{len(result)} Paths using {algo.upper()}")
            result_window.geometry("600x400")

            output_text = tk.Text(result_window, wrap=tk.WORD, font=("Courier", 10))
            output_text.pack(expand=True, fill=tk.BOTH)

            output_text.insert(
                tk.END, f"✅ Found {len(result)} paths using Top-K + {algo.upper()}\n\n"
            )

            for idx, (cost, path) in enumerate(result):
                path_str = " → ".join(str(node) for node in path)
                output_text.insert(
                    tk.END,
                    f"Path {idx+1}:\n{path_str}\n🕒 Travel Time: {cost:.2f} minutes\n\n",
                )
                color = color_list[idx % len(color_list)]
                self.graph.highlight_path(self.canvas, path, color=color)
        else:
            path, cost = result
            self.graph.highlight_path(self.canvas, path, color="red")

            result_window = tk.Toplevel(self.root)
            result_window.title(f"Path using {algo.upper()}")
            result_window.geometry("500x200")

            output_text = tk.Text(result_window, wrap=tk.WORD, font=("Courier", 10))
            output_text.pack(expand=True, fill=tk.BOTH)

            output_text.insert(tk.END, f"✅ Path found using {algo.upper()}\n\n")
            path_str = " → ".join(str(node) for node in path)
            output_text.insert(
                output_text.index(tk.END),
                f"{path_str}\n🕒 Travel Time: {cost:.2f} minutes\n",
            )
            output_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()
