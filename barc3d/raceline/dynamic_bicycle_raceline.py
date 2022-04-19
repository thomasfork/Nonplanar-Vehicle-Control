''' nonplanar dynamic bicycle raceline '''
from barc3d.surfaces import load_surface
from barc3d.pytypes import VehicleConfig
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.dynamics.dynamics_3d import DynamicsModel
from barc3d.raceline.base_raceline import BaseRaceline, RacelineConfig
from barc3d.dynamics.dynamics_3d import DynamicBicycle3D

from barc3d.raceline.kinematic_bicycle_raceline import KinematicBicycleRaceline

class DynamicBicycleRaceline(BaseRaceline):
    '''
    Raceline using a dynamic bicycle model with steering angle and acceleration inputs
    Enforces normal force constraints and longitudinal friction constraint
    '''

    def _get_raceline_label(self):
        return 'Dynamic Bicycle'

    def _get_raceline_color(self):
        return [0,1,0,1]

    def _get_model(self, vehicle_config: VehicleConfig, surf: BaseSurface) -> DynamicsModel:
        return DynamicBicycle3D(vehicle_config = vehicle_config, surf = surf)

    def _initial_setup_checks(self):
        assert isinstance(self.model, DynamicBicycle3D)
        assert self.model.use_dae is False
        self.is_dae = False

