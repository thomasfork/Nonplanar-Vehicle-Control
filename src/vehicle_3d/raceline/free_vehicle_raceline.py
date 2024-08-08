'''
free (not tangent) two track slip input raceline
'''
import numpy as np

from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.dynamics_model import DynamicsModel
from vehicle_3d.models.free_vehicle_model import FreeSlipInputVehicleModel, FreeVehicleModelConfig
from vehicle_3d.utils.ocp_util import OCPResults
from vehicle_3d.raceline.base_raceline import BaseRaceline, BasePlanarRaceline, RacelineConfig

class FreeSlipInputRaceline(BaseRaceline):
    ''' free two track slip input raceline '''
    model_config: FreeVehicleModelConfig
    def __init__(self,
                 surf: BaseSurface,
                 config: RacelineConfig,
                 model_config: FreeVehicleModelConfig,
                 ws_raceline: OCPResults = None,
                 ws_model: DynamicsModel = None):
        super().__init__(surf, config, model_config, ws_raceline, ws_model)

    def _get_model(self) -> FreeSlipInputVehicleModel:
        return FreeSlipInputVehicleModel(self.model_config, self.surf)

    def _get_state_bounds(self,n,k):
        state_u, state_l = super()._get_state_bounds(n,k)
        state_u[3] = np.pi
        state_l[3] =-np.pi
        state_l[6] = 0
        return state_u, state_l

    def _enforce_initial_constraints(self):
        z0 = self.ocp_vars.Z[0,0]
        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        if self.config.y0 is not None:
            g += [z0[1] - self.config.y0]
            ubg += [0.]
            lbg += [0.]
        if self.config.v0 is not None:
            g += [z0[-6] - self.config.v0, z0[-5], z0[-4]]
            ubg += [0.] * 3
            lbg += [0.] * 3

    def _guess_z(self,n,k):
        z = super()._guess_z(n,k)
        z[3] = self.model_config.h
        z[6] = self.config.v_ws
        return z

class PlanarFreeSlipInputRaceline(FreeSlipInputRaceline, BasePlanarRaceline):
    ''' planar free two track slip input raceline '''
