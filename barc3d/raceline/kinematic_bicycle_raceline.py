''' nonplanar kinematic bicycle raceline'''
from barc3d.surfaces import load_surface
from barc3d.pytypes import VehicleConfig
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.dynamics.dynamics_3d import DynamicsModel
from barc3d.raceline.base_raceline import BaseRaceline, RacelineConfig
from barc3d.dynamics.dynamics_3d import KinematicBicycle3D

class KinematicBicycleRaceline(BaseRaceline):
    '''
    Raceline using a kinematic bicycle model with steering angle and acceleration inputs
    Enforces normal force constraints and a friction cone constraint.
    '''

    def _get_raceline_label(self):
        return 'Kinematic Bicycle'

    def _get_raceline_color(self):
        return [0,0,1,1]

    def _get_model(self, vehicle_config: VehicleConfig, surf: BaseSurface) -> DynamicsModel:
        return KinematicBicycle3D(vehicle_config = vehicle_config, surf = surf)

    def _initial_setup_checks(self):
        assert isinstance(self.model, KinematicBicycle3D)
        assert self.model.use_dae is False
        self.is_dae = False

