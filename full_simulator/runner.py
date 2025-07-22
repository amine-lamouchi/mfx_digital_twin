from lcls_beamline_toolbox.models import mfx
from alignment_interface import UndulatorPointingTask
from simple_optimizer import random_search, grid_search, bayesian_optimization

def main():
    # set up the simulator
    mfx_sim = mfx.MFX(E0=9000, N=256)
    mfx_sim.propagate()

    # create the alignment task
    task = UndulatorPointingTask(mfx_sim)

    # run the optimizer
    best_dofs, best_cost = bayesian_optimization(task, n_calls=30)
    print("Best DoFs:", best_dofs)
    print("Minimum Cost:", best_cost)
    print("--------------------------------")

if __name__ == "__main__":
    main()
