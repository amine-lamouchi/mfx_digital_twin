import numpy as np

def random_search(task, n_trials=100):
    """Simple random optimizer to test the task interface."""
    bounds = task.get_bounds()
    best_dofs = None
    best_cost = float("inf")

    for _ in range(n_trials):
        candidate = [np.random.uniform(low, high) for (low, high) in bounds]
        task.set_dofs(candidate)
        cost = task.evaluate_objective()

        print(f"Random trial {_} DoFs: {candidate}")
        print(f"Random trial {_} cost: {cost}")
        print("--------------------------------")
        
        if cost < best_cost:
            best_cost = cost
            best_dofs = candidate

    return best_dofs, best_cost


def grid_search(task, points_per_dim=5):
    """Grid search optimizer for the task interface. Tries all combinations of evenly spaced points in each dimension."""
    bounds = task.get_bounds()
    grid_axes = [np.linspace(low, high, points_per_dim) for (low, high) in bounds]
    mesh = np.meshgrid(*grid_axes)
    grid_points = np.stack([m.flatten() for m in mesh], axis=-1)
    best_dofs = None
    best_cost = float("inf")
    for i, candidate in enumerate(grid_points):
        task.set_dofs(candidate)
        cost = task.evaluate_objective()
        print(f"Grid trial {i} DoFs: {candidate}")
        print(f"Grid trial {i} cost: {cost}")
        print("--------------------------------")
        if cost < best_cost:
            best_cost = cost
            best_dofs = candidate
    return best_dofs, best_cost


