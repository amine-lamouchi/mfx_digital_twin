import numpy as np

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

