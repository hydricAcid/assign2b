import os
import sys
import tkinter as tk
from tkinter import messagebox

# Đảm bảo import được module trong thư mục main
sys.path.append(os.path.abspath("main"))

from graph import Graph
from search import SearchAlgorithmExecutor


class TrafficApp:
    def __init__(self, master):
        self.master = master
        master.title("Traffic Route Finder")

        # Nhập start
        tk.Label(master, text="Start SCATS ID:").grid(row=0, column=0, padx=10, pady=5)
        self.start_entry = tk.Entry(master)
        self.start_entry.grid(row=0, column=1)

        # Nhập goal
        tk.Label(master, text="Goal SCATS ID:").grid(row=1, column=0, padx=10, pady=5)
        self.goal_entry = tk.Entry(master)
        self.goal_entry.grid(row=1, column=1)

        # Chọn thuật toán
        tk.Label(master, text="Algorithm:").grid(row=2, column=0, padx=10, pady=5)
        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("A_STAR")  # Mặc định
        algorithm_options = ["A_STAR", "BFS", "DFS", "GBFS", "CUSTOM1", "CUSTOM2"]
        self.algorithm_menu = tk.OptionMenu(
            master, self.algorithm_var, *algorithm_options
        )
        self.algorithm_menu.grid(row=2, column=1)

        # Nút chạy
        self.find_button = tk.Button(master, text="Find Path", command=self.find_path)
        self.find_button.grid(row=3, column=0, columnspan=2, pady=10)

    def find_path(self):
        start = self.start_entry.get().strip()
        goal = self.goal_entry.get().strip()
        algorithm = self.algorithm_var.get()

        if not start or not goal:
            messagebox.showwarning(
                "Missing Input", "Please enter both start and goal SCATS IDs."
            )
            return

        graph = Graph("data/weighted_edges.txt")

        if not graph.has_node(start):
            messagebox.showerror("Invalid Node", f"Start node {start} does not exist.")
            return

        if not graph.has_node(goal):
            messagebox.showerror("Invalid Node", f"Goal node {goal} does not exist.")
            return

        executor = SearchAlgorithmExecutor(algorithm, graph)
        path, cost = executor.search(start, goal)

        if not path:
            messagebox.showinfo("Result", f"No path found from {start} to {goal}.")
        else:
            route_str = " → ".join(path)
            messagebox.showinfo(
                "Route Found",
                f"Path: {route_str}\nTotal travel time: {cost:.2f} minutes",
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()
