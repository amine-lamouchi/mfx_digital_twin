import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from alignment_interface import (
    UndulatorPointingTask, 
    BeamSteeringTask, 
    TransfocatorTask, 
    VonHamosTask
)
from lcls_beamline_toolbox.models import mfx


class ScanGenerator:
    """Generates scans for each degree of freedom of each alignment task."""
    
    def __init__(self, output_dir="scans", n_points=20):
        """
        Initialize the scan generator.
        
        Args:
            output_dir: Directory to save scan results
            n_points: Number of points to scan for each degree of freedom
        """
        self.output_dir = output_dir
        self.n_points = n_points
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing MFX simulation...")
        self.sim = mfx.MFX(E0=9000, N=256)
        self.sim.propagate()
        
        self.tasks = {
            "undulator_pointing": UndulatorPointingTask(self.sim),
            "beam_steering": BeamSteeringTask(self.sim),
            "transfocator": TransfocatorTask(self.sim),
            "vonhamos": VonHamosTask(self.sim)
        }
        
        self.default_values = {}
        for task_name, task in self.tasks.items():
            self.default_values[task_name] = task.get_dofs().copy()
            print(f"Stored default values for {task_name}: {self.default_values[task_name]}")


    def _get_dof_labels(self, task_name, task):
        """Return a list of human-readable labels for the task's DoFs."""
        if task_name == "undulator_pointing":
            return ["ax", "ay"]
        elif task_name == "beam_steering":
            return ["pitch"]
        elif task_name == "transfocator":
            labels = []
            if hasattr(task, "enabled_crls"):
                for i, _ in task.enabled_crls:
                    labels.append(f"crl{i}_x")
                    labels.append(f"crl{i}_y")
            return labels
        elif task_name == "vonhamos":
            return ["pitch", "yaw", "y"]
        else:
            return [f"dof_{i}" for i in range(len(task.get_bounds()))]
    
    def _reinitialize_simulation(self):
        """Reinitialize the simulation to ensure clean state."""
        print("Reinitializing simulation...")
        self.sim = mfx.MFX(E0=9000, N=256)
        self.sim.propagate()
        
    def scan_single_dof(self, task_name, dof_index, task):
        """
        Scan a single degree of freedom while keeping others at default values.
        
        Args:
            task_name: Name of the task
            dof_index: Index of the degree of freedom to scan
            task: The alignment task object
        """
        print(f"Scanning {task_name} DOF {dof_index}...")
        
        task_dir = os.path.join(self.output_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        
        bounds = task.get_bounds()
        if dof_index >= len(bounds):
            print(f"DOF index {dof_index} out of range for {task_name}")
            return
            
        dof_bounds = bounds[dof_index]
        
        default_dofs = self.default_values[task_name].copy()
        
        scan_values = np.linspace(dof_bounds[0], dof_bounds[1], self.n_points)
        
        dof_labels = self._get_dof_labels(task_name, task)
        label = dof_labels[dof_index] if dof_index < len(dof_labels) else f"dof_{dof_index}"

        for i, value in enumerate(scan_values):
            new_dofs = default_dofs.copy()
            new_dofs[dof_index] = value
            
            task.set_dofs(new_dofs)
            cost = task.evaluate_objective()
            
            unique_id = f"{label}_step_{i:03d}"
            task.save_diagnostic(unique_id, task_dir)
            
            print(f"  Step {i}: value={value:.6e}, cost={cost:.6e}")
        
        print(f"  Resetting {task_name} to original default values...")
        task.set_dofs(self.default_values[task_name])
    
    def create_gif(self, task_name, dof_index, task):
        """
        Create a GIF from the diagnostic images for a specific DOF scan.
        
        Args:
            task_name: Name of the task
            dof_index: Index of the degree of freedom
            task: The alignment task object
        """
        task_dir = os.path.join(self.output_dir, task_name)

        dof_labels = self._get_dof_labels(task_name, task)
        label = dof_labels[dof_index] if dof_index < len(dof_labels) else f"dof_{dof_index}"

        if task_name == "undulator_pointing":
            pattern = os.path.join(task_dir, f"hx2_shared_{label}_step_*.png")
        elif task_name == "beam_steering":
            pattern = os.path.join(task_dir, f"DG1_YAG_{label}_step_*.png")
        elif task_name == "transfocator":
            pattern = os.path.join(task_dir, f"DG2_YAG_{label}_step_*.png")
        elif task_name == "vonhamos":
            pattern = os.path.join(task_dir, f"vonhamos_{label}_step_*.png")
        else:
            print(f"Unknown task name: {task_name}")
            return
        
        image_files = sorted(glob.glob(pattern))
        
        if not image_files:
            print(f"No images found for {task_name} DOF {dof_index}")
            print(f"Pattern searched: {pattern}")
            return
        
        images = []
        for img_file in image_files:
            try:
                img = Image.open(img_file)
                images.append(img)
            except Exception as e:
                print(f"Failed to load {img_file}: {e}")
        
        if not images:
            print(f"No valid images found for {task_name} DOF {dof_index}")
            return
        
        gif_path = os.path.join(task_dir, f"{label}_scan.gif")
        try:
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=500,  
                loop=0
            )
            print(f"Created GIF: {gif_path}")
            try:
                for png in image_files:
                    os.remove(png)
            except Exception as del_err:
                print(f"Warning: could not delete some PNGs for {task_name} {label}: {del_err}")
        except Exception as e:
            print(f"Failed to create GIF for {task_name} DOF {dof_index}: {e}")
    
    def generate_all_scans(self):
        """Generate scans for all tasks and all degrees of freedom."""
        print("Starting comprehensive scan generation...")
        
        for task_name, task in self.tasks.items():
            print(f"\n{'='*50}")
            print(f"Processing task: {task_name}")
            print(f"{'='*50}")
            
            self._reinitialize_simulation()
            
            if task_name == "undulator_pointing":
                task = UndulatorPointingTask(self.sim)
            elif task_name == "beam_steering":
                task = BeamSteeringTask(self.sim)
            elif task_name == "transfocator":
                task = TransfocatorTask(self.sim)
            elif task_name == "vonhamos":
                task = VonHamosTask(self.sim)
            
            self.default_values[task_name] = task.get_dofs().copy()
            print(f"Updated default values for {task_name}: {self.default_values[task_name]}")
            
            bounds = task.get_bounds()
            n_dofs = len(bounds)
            
            print(f"Task has {n_dofs} degrees of freedom")
            
            for dof_index in range(n_dofs):
                print(f"\n--- Scanning DOF {dof_index} ---")
                self.scan_single_dof(task_name, dof_index, task)
                self.create_gif(task_name, dof_index, task)
        
        print(f"\n{'='*50}")
        print("Scan generation complete!")
        print(f"Results saved in: {self.output_dir}")
        print(f"{'='*50}")

def main():
    """Main function to run the scan generator."""
    generator = ScanGenerator(output_dir="scans", n_points=20)
    
    generator.generate_all_scans()

if __name__ == "__main__":
    main() 