from lcls_beamline_toolbox.models import mfx
from alignment_interface import UndulatorPointingTask, BeamSteeringTask, TransfocatorTask, VonHamosTask
from simple_optimizer import random_search, grid_search, bayesian_optimization
from history_logger import HistoryLogger
from utils import make_experiment_dir

def main():
    # set up the simulator
    print("Setting up MFX simulator...")
    mfx_sim = mfx.MFX(E0=9000, N=256)
    mfx_sim.propagate()
    print("MFX simulator ready!")

    # choose the task and optimizer
    task = VonHamosTask(mfx_sim)
    task_name = "vonhamos"
    optimizer_name = "bayesian"

    # set up experiment logging
    outdir = make_experiment_dir(task_name, optimizer_name)
    logger = HistoryLogger(outdir, save_images=True, real_time_csv=True)
    print(f"Logging to: {outdir}")

    # run the optimizer with logging
    print(f"Starting {optimizer_name} optimization...")
    best_dofs, best_cost = bayesian_optimization(task, n_calls=30, logger=logger)

    # print results and summary
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print("Best DoFs:", best_dofs)
    print("Minimum Cost:", best_cost)
    
    # Print detailed summary from logger
    logger.print_summary()
    print("="*50)

if __name__ == "__main__":
    main()
