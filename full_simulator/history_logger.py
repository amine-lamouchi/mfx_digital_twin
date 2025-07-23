import os
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

class HistoryLogger:
    def __init__(self, output_dir, save_images=True, real_time_csv=True, task_name=None, method_name=None):
        self.output_dir = output_dir
        self.save_images = save_images
        self.real_time_csv = real_time_csv
        self.history = []
        self.header_written = False
        self.task_name = task_name
        self.method_name = method_name
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "history.csv")
        
        # Initialize CSV file with header if real-time writing is enabled
        if self.real_time_csv:
            self._write_csv_header()

    def _write_csv_header(self):
        """Write CSV header to file."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # We'll write the header when we have the first entry
            pass

    def log(self, index, candidate, cost, task=None):
        """Log a trial result."""
        entry = {"trial": index, "cost": cost}
        for i, val in enumerate(candidate):
            entry[f"dof_{i}"] = val
        self.history.append(entry)

        # Write to CSV in real-time if enabled
        if self.real_time_csv:
            self._append_to_csv(entry)

        # Save diagnostic image if supported
        if self.save_images and task and hasattr(task, "save_diagnostic"):
            try:
                task.save_diagnostic(index, self.output_dir)
                print(f"[HistoryLogger] Saved diagnostic image for trial {index}")
            except Exception as e:
                print(f"[HistoryLogger] Failed to save diagnostic for trial {index}: {e}")

    def _append_to_csv(self, entry):
        """Append a single entry to the CSV file."""
        try:
            # Write header if this is the first entry
            if not self.header_written:
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=entry.keys())
                    writer.writeheader()
                self.header_written = True
            
            # Append the entry
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=entry.keys())
                writer.writerow(entry)
        except Exception as e:
            print(f"[HistoryLogger] Failed to write to CSV: {e}")

    def save_csv(self):
        if not self.history:
            print("[HistoryLogger] No history to save")
            return
            
        try:
            keys = self.history[0].keys()
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)
            print(f"[HistoryLogger] Saved {len(self.history)} trials to {self.csv_path}")
        except Exception as e:
            print(f"[HistoryLogger] Failed to save CSV: {e}")

    def plot_cost_history(self, filename="history_plot.png", task_name=None, method_name=None, penalty_threshold=1e6):
        if not self.history:
            print("[HistoryLogger] No history to plot")
            return
            
        try:
            costs = [entry["cost"] for entry in self.history]
            trials = [entry["trial"] for entry in self.history]

            # Filter out penalty values for scatter plot
            scatter_trials = [t for t, c in zip(trials, costs) if c < penalty_threshold and np.isfinite(c)]
            scatter_costs  = [c for c in costs if c < penalty_threshold and np.isfinite(c)]

            # Compute best-so-far ignoring penalty values
            best_so_far = []
            current_best = np.inf
            for c in costs:
                if c < penalty_threshold and np.isfinite(c):
                    current_best = min(current_best, c)
                best_so_far.append(current_best if np.isfinite(current_best) else np.nan)

            plt.figure(figsize=(10, 6))
            plt.scatter(scatter_trials, scatter_costs, color="blue", marker='o', s=20, label="Cost")
            plt.plot(trials, best_so_far, color="green", linewidth=1, linestyle="--", label="Best so far")
            plt.xlabel("Trial")
            plt.ylabel("Cost")

            # Manage x-axis ticks to avoid overcrowding
            ax = plt.gca()
            if len(trials) <= 15:
                ax.set_xticks(trials)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
                plt.xticks(rotation=45)

            # Manage y-axis ticks for readability
            ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

            plot_task_name = task_name if task_name is not None else self.task_name
            plot_method_name = method_name if method_name is not None else self.method_name

            if plot_task_name and plot_method_name:
                plt.title(f"{plot_task_name} - {plot_method_name}")
            elif plot_task_name:
                plt.title(f"{plot_task_name}")
            elif plot_method_name:
                plt.title(f"{plot_method_name}")
            else:
                plt.title("Optimization Cost over Trials")

            plt.grid(False)
            plt.legend()
            plt.tight_layout()
            
            plot_path = os.path.join(self.output_dir, filename)
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[HistoryLogger] Saved cost history plot to {plot_path}")
        except Exception as e:
            print(f"[HistoryLogger] Failed to create plot: {e}")

    def get_best_result(self):
        """Get the best result from the history."""
        if not self.history:
            return None, None
            
        best_entry = min(self.history, key=lambda x: x["cost"])
        best_cost = best_entry["cost"]
        best_dofs = [best_entry[f"dof_{i}"] for i in range(len(best_entry) - 2)]  # -2 for trial and cost
        return best_dofs, best_cost

    def print_summary(self):
        """Print a summary of the optimization results."""
        if not self.history:
            print("[HistoryLogger] No history to summarize")
            return
            
        best_dofs, best_cost = self.get_best_result()
        print(f"\n[HistoryLogger] Optimization Summary:")
        print(f"  Total trials: {len(self.history)}")
        print(f"  Best cost: {best_cost}")
        print(f"  Best DoFs: {best_dofs}")
        print(f"  Output directory: {self.output_dir}")
