import numpy as np
from alignment_interface import LARGE_PENALTY
import scipy.optimize

def random_search(task, n_trials=100, logger=None):
    """Simple random optimizer to test the task interface."""
    bounds = task.get_bounds()
    best_dofs = None
    best_cost = float("inf")

    for i in range(n_trials):
        candidate = [np.random.uniform(low, high) for (low, high) in bounds]
        try:
            task.set_dofs(candidate)
            cost = task.evaluate_objective()
        except Exception as e:
            print(f"Random search trial {i} failed: {e}")
            cost = LARGE_PENALTY

        print(f"Random search trial {i} DoFs: {candidate}")
        print(f"Random search trial {i} cost: {cost}")
        print("--------------------------------")

        if logger:
            logger.log(i, candidate, cost, task)
        
        if cost < best_cost:
            best_cost = cost
            best_dofs = candidate

    if logger:
        logger.save_csv()
        logger.plot_cost_history()

    return best_dofs, best_cost


def grid_search(task, points_per_dim=5, logger=None):
    """Grid search optimizer for the task interface. Tries all combinations of evenly spaced points in each dimension."""
    bounds = task.get_bounds()
    grid_axes = [np.linspace(low, high, points_per_dim) for (low, high) in bounds]
    mesh = np.meshgrid(*grid_axes)
    grid_points = np.stack([m.flatten() for m in mesh], axis=-1)
    best_dofs = None
    best_cost = float("inf")
    for i, candidate in enumerate(grid_points):
        try:
            task.set_dofs(candidate)
            cost = task.evaluate_objective()
        except Exception as e:
            print(f"Grid search trial {i} failed: {e}")
            cost = LARGE_PENALTY
        print(f"Grid search trial {i} DoFs: {candidate}")
        print(f"Grid search trial {i} cost: {cost}")
        print("--------------------------------")
        
        if logger:
            logger.log(i, candidate, cost, task)
            
        if cost < best_cost:
            best_cost = cost
            best_dofs = candidate

    if logger:
        logger.save_csv()
        logger.plot_cost_history()

    return best_dofs, best_cost


def bayesian_optimization(task, n_calls=30, logger=None):
    """Bayesian optimization using scikit-optimize (skopt) for the task interface."""
    try:
        from skopt import Optimizer
    except ImportError:
        raise ImportError("scikit-optimize (skopt) is required for bayesian_optimization. Install with 'pip install scikit-optimize'.")
    bounds = task.get_bounds()
    opt = Optimizer(dimensions=bounds, random_state=0)
    best_dofs = None
    best_cost = float("inf")
    for i in range(n_calls):
        candidate = opt.ask()
        try:
            task.set_dofs(candidate)
            cost = task.evaluate_objective()
        except Exception as e:
            print(f"Bayesian optimization trial {i} failed: {e}")
            cost = LARGE_PENALTY
        opt.tell(candidate, cost)
        print(f"Bayesian optimization trial {i} DoFs: {candidate}")
        print(f"Bayesian optimization trial {i} cost: {cost}")
        print("--------------------------------")
        
        if logger:
            logger.log(i, candidate, cost, task)
            
        if cost < best_cost:
            best_cost = cost
            best_dofs = candidate

    if logger:
        logger.save_csv()
        logger.plot_cost_history()

    return best_dofs, best_cost


