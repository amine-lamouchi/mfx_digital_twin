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
        self.sim.propagate()

    def get_observation(self):
        cx = self.sim.beamline.hx2_shared.cx
        cy = self.sim.beamline.hx2_shared.cy
        return [cx, cy]

    def evaluate_objective(self):
        cx, cy = self.get_observation()
        return np.sqrt(cx**2 + cy**2)

    def get_bounds(self):
        return [(-2.5e-6, 2.5e-6), (-2.5e-6, 2.5e-6)]

class BeamSteeringTask(AlignmentTask):
    def get_dofs(self):
        return [self.sim.mr1l4_pitch.wm()]

    def set_dofs(self, values):
        self.sim.mr1l4_pitch.mvr(values[0])
        self.sim.propagate()

    def get_observation(self):
        cx = self.sim.beamline.DG1_YAG.cx
        return cx
    
    def evaluate_objective(self):
        cx = self.get_observation()
        return np.abs(cx)
    
    def get_bounds(self):
        return [(-5e-6, 5e-6)]
