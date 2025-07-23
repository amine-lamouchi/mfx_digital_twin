from lcls_beamline_toolbox.models import mfx
from alignment_interface import UndulatorPointingTask, BeamSteeringTask, TransfocatorTask, VonHamosTask
from simple_optimizer import random_search, grid_search, bayesian_optimization
from history_logger import HistoryLogger
from utils import make_experiment_dir

def main():

    tasks = [
        (UndulatorPointingTask, "Undulator Pointing"),
        (BeamSteeringTask, "Beam Steering"),
        (TransfocatorTask, "Transfocator"),
        (VonHamosTask, "Von Hamos Alignment")
    ]

    optimizer_name = "Bayesian Optimization"

    for TaskClass, task_name in tasks:

        print(f"\n{'='*60}\nRunning {task_name} with {optimizer_name}\n{'='*60}")

        mfx_sim = mfx.MFX(E0=9000, N=256)
        mfx_sim.propagate()
        task = TaskClass(mfx_sim)
        outdir = make_experiment_dir(task_name, optimizer_name)
        logger = HistoryLogger(outdir, save_images=False, real_time_csv=True, task_name=task_name, method_name=optimizer_name)
        print(f"Logging to: {outdir}")
        
        best_dofs, best_cost = bayesian_optimization(task, n_calls=100, logger=logger)
        print("\n" + "="*50)
        print("OPTIMIZATION COMPLETE")
        print("="*50)
        print("Best DoFs:", best_dofs)
        print("Minimum Cost:", best_cost)
        logger.print_summary()
        print("="*50)

        task.set_dofs(best_dofs)
        task.save_diagnostic("best", outdir)

if __name__ == "__main__":
    main()
