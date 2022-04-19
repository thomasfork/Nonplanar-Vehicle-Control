''' nonplanar two track slip input raceline '''
import numpy as np

from barc3d.surfaces import load_surface
from barc3d.pytypes import VehicleConfig
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.dynamics.dynamics_3d import DynamicsModel
from barc3d.raceline.base_raceline import BaseRaceline, RacelineConfig
from barc3d.dynamics.dynamics_3d import DynamicTwoTrackSlipInput3D


class DynamicTwoTrackSlipInputRaceline(BaseRaceline):
    '''
    Raceline using a two track model with steering angle and tire slip inputs
    Enforces normal force constraints on individual tires.
    '''

    def _get_raceline_label(self):
        return 'Dynamic Two Track'

    def _get_raceline_color(self):
        return [1,0,0,1]

    def _get_model(self, vehicle_config: VehicleConfig, surf: BaseSurface) -> DynamicsModel:
        return DynamicTwoTrackSlipInput3D(vehicle_config = vehicle_config, surf = surf)

    def _initial_setup_checks(self):
        assert isinstance(self.model, DynamicTwoTrackSlipInput3D)
        assert self.model.use_dae is True
        self.is_dae = True

