import os
import csv
import matplotlib.pyplot as plt
import numpy as np

class HistoryLogger:
    def __init__(self, output_dir, save_images=True, real_time_csv=True):
        self.output_dir = output_dir
        self.save_images = save_images
        self.real_time_csv = real_time_csv
        self.history = []
        self.header_written = False
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
        """Save all history to CSV file (for non-real-time mode)."""
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

    def plot_cost_history(self, filename="history_plot.png"):
        """Plot the cost history."""
        if not self.history:
            print("[HistoryLogger] No history to plot")
            return
            
        try:
            costs = [entry["cost"] for entry in self.history]
            trials = [entry["trial"] for entry in self.history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(trials, costs, marker='o', linestyle='-', markersize=4)
            plt.xlabel("Trial")
            plt.ylabel("Cost")
            plt.title("Optimization Cost over Trials")
            plt.grid(True, alpha=0.3)
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
