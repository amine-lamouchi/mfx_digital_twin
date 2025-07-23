import numpy as np
import vonhamos_spectrometer as vh

class AlignmentTask:
    """Base class for all alignment tasks."""
    def __init__(self, sim):
        self.sim = sim

    def get_dofs(self):
        raise NotImplementedError

    def set_dofs(self, values):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def evaluate_objective(self):
        raise NotImplementedError

    def get_bounds(self):
        raise NotImplementedError


class UndulatorPointingTask(AlignmentTask):
    def get_dofs(self):
        ax = self.sim.beam_params['ax']
        ay = self.sim.beam_params['ay']
        return [ax, ay]

    def set_dofs(self, values):
        ax, ay = values
        self.sim.undulator_pointing(ax=ax, ay=ay)
        try:
            self.sim.propagate()
        except Exception as e:
            print(f"[UndulatorPointingTask] propagate() failed: {e}")
            self._propagation_failed = True
        else:
            self._propagation_failed = False


    def get_observation(self):
        cx = self.sim.beamline.hx2_shared.cx
        cy = self.sim.beamline.hx2_shared.cy
        return [cx, cy]

    def evaluate_objective(self):
        if getattr(self, "_propagation_failed", False):
            return float("inf")     
        cx, cy = self.get_observation()
        return np.sqrt(cx**2 + cy**2)

    def get_bounds(self):
        return [(-2.5e-6, 2.5e-6), (-2.5e-6, 2.5e-6)]
    
    def save_diagnostic(self, index, output_dir):
        fig = self.sim.beamline.hx2_shared.view_beam()
        fig.savefig(f"{output_dir}/hx2_shared_{index}.png")


class BeamSteeringTask(AlignmentTask):
    def get_dofs(self):
        return [self.sim.mr1l4_pitch.wm()]

    def set_dofs(self, values):
        self.sim.mr1l4_pitch.mvr(values[0])
        try:
            self.sim.propagate()
        except Exception as e:
            print(f"[BeamSteeringTask] propagate() failed: {e}")
            self._propagation_failed = True
        else:
            self._propagation_failed = False

    def get_observation(self):
        cx = self.sim.beamline.DG1_YAG.cx
        return cx
    
    def evaluate_objective(self):
        if getattr(self, "_propagation_failed", False):
            return float("inf")     
        cx = self.get_observation()
        return np.abs(cx)
    
    def get_bounds(self):
        return [(-5e-6, 5e-6)]
    
    def save_diagnostic(self, index, output_dir):
        fig = self.sim.beamline.DG1_YAG.view_beam()
        fig.savefig(f"{output_dir}/DG1_YAG_{index}.png")


class TransfocatorTask(AlignmentTask):
    def __init__(self, sim):
        super().__init__(sim)
        self.enabled_crls = [
            (i, crl)
            for i, crl in enumerate(sim.tfs_list, start=2) # start at 2 because mfx model indexes the crls from 2 to 10
            if crl.enabled
        ]
        self.n_crls = len(self.enabled_crls)

    def get_dofs(self):
        dofs = []
        for i, _ in self.enabled_crls:
            dofs.append(self.sim.__getattr__(f"tfs_{i}_x").wm())
            dofs.append(self.sim.__getattr__(f"tfs_{i}_y").wm())
        return dofs

    def set_dofs(self, values):
        for idx, (i, _) in enumerate(self.enabled_crls):
            getattr(self.sim, f"tfs_{i}_x").mvr(values[2*idx])
            getattr(self.sim, f"tfs_{i}_y").mvr(values[2*idx + 1])
        try:
            self.sim.propagate()
        except Exception as e:
            print(f"[TransfocatorTask] propagate() failed: {e}")
            self._propagation_failed = True
        else:
            self._propagation_failed = False

    def get_observation(self):
        cx = self.sim.beamline.DG2_YAG.cx
        cy = self.sim.beamline.DG2_YAG.cy
        wx = self.sim.beamline.DG2_YAG.wx
        wy = self.sim.beamline.DG2_YAG.wy
        return cx, cy, wx, wy

    def evaluate_objective(self):
        if getattr(self, "_propagation_failed", False):
            return float("inf")     
        cx, cy, wx, wy = self.get_observation()
        return np.sqrt(cx**2 + cy**2) + np.abs(wx - wy)

    def get_bounds(self):
        return [(-5e-6, 5e-6)] * 2 * self.n_crls
    
    def save_diagnostic(self, index, output_dir):
        fig = self.sim.beamline.DG2_YAG.view_beam()
        fig.savefig(f"{output_dir}/DG2_YAG_{index}.png")


class VonHamosTask(AlignmentTask):
    def __init__(self, sim,
                 pitch_bounds=(-np.deg2rad(1), np.deg2rad(1)), # +- 1 degree for pitch
                 yaw_bounds=(-np.deg2rad(5), np.deg2rad(5)), # +- 5 degree for yaw
                 y_bounds=(-5.0, 5.0)): # +- 5 mm for translation
        super().__init__(sim)
        self.sim.propagate()
        ip = sim.beamline.MFX_IP
        self.beamH = ip.wx * 1000 / 2.355
        self.beamV = ip.wy * 1000 / 2.355
        baseBL = vh.build_beamline(energy=sim.E0, beamH=self.beamH, beamV=self.beamV)
        self.center0 = baseBL.vonHamos01.center
        self.pitch0 = float(baseBL.vonHamos01.pitch)
        self.yaw0 = float(getattr(baseBL.vonHamos01, "yaw", 0.0))
        self.bounds = [
            (self.pitch0 + pitch_bounds[0], self.pitch0 + pitch_bounds[1]),
            (self.yaw0   + yaw_bounds[0],   self.yaw0   + yaw_bounds[1]),
            (self.center0[1] + y_bounds[0], self.center0[1] + y_bounds[1])
        ]
        self.last_plot = None
        self._propagation_failed = False
        self.last_vals = [self.pitch0, self.yaw0, self.center0[1]]

    def get_dofs(self):
        return list(self.last_vals)

    def set_dofs(self, values):
        pitch_val, yaw_val, y_val = values
        bl = vh.build_beamline(energy=self.sim.E0, beamH=self.beamH, beamV=self.beamV)
        bl.vonHamos01.pitch = pitch_val
        setattr(bl.vonHamos01, "yaw", yaw_val)
        cx0, _, cz0 = self.center0
        bl.vonHamos01.center = (cx0, y_val, cz0)
        try:
            vh.rrun.run_process = vh.run_process
            plots = vh.define_plots()
            vh.xrtrun.run_ray_tracing(plots=plots, beamLine=bl, backend="raycing")
            self.last_plot = plots[0]
            self._propagation_failed = False
        except Exception as e:
            print(f"[VonHamosTask] ray tracing failed: {e}")
            self._propagation_failed = True
            self.last_plot = None
        self.last_vals = values

    def get_observation(self):
        if self._propagation_failed or self.last_plot is None:
            return None
        cx = self.last_plot.cx
        cy = self.last_plot.cy
        dx = self.last_plot.dx / 2
        dy = self.last_plot.dy / 2
        return cx, cy, dx, dy

    def evaluate_objective(self):
        if self._propagation_failed:
            return float("inf")
        cx, cy, dx, dy = self.get_observation()
        return np.sqrt(cx**2 + cy**2) + abs(dx)

    def get_bounds(self):
        return self.bounds
    
    def save_diagnostic(self, index, output_dir):
        if self.last_plot is not None and not self._propagation_failed:
            try:
                self.last_plot.saveName = f"{output_dir}/vonhamos_{index}"
                self.last_plot.save()
            except Exception as e:
                print(f"[VonHamosTask] Failed to save diagnostic plot for trial {index}: {e}")
        else:
            print(f"[VonHamosTask] No diagnostic plot available for trial {index} (ray tracing failed)")