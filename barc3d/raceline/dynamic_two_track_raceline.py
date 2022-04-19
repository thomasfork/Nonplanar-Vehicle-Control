''' nonplanar two track raceline '''
from barc3d.surfaces import load_surface
from barc3d.pytypes import VehicleConfig
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.dynamics.dynamics_3d import DynamicsModel
from barc3d.raceline.base_raceline import BaseRaceline, RacelineConfig
from barc3d.dynamics.dynamics_3d import DynamicTwoTrack3D

from barc3d.raceline.dynamic_bicycle_raceline import DynamicBicycleRaceline

class DynamicTwoTrackRaceline(BaseRaceline):
    '''
    Raceline using a two track model based on acceleration/steering input
    Enforces normal force constraints on individual tires.
    ***WARNING*** Does not solve raceline reliably, use slip input version.
    '''

    def _get_raceline_label(self):
        return ''

    def _get_raceline_color(self):
        return [1,0,1,1]

    def _get_model(self, vehicle_config: VehicleConfig, surf: BaseSurface) -> DynamicsModel:
        return DynamicTwoTrack3D(vehicle_config = vehicle_config, surf = surf, use_dae = True)

    def _initial_setup_checks(self):
        assert isinstance(self.model, DynamicTwoTrack3D)
        assert self.model.use_dae is True
        self.is_dae = True

