'''
kinematic bicycle raceline
'''
import numpy as np
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.dynamics_model import DynamicsModel
from vehicle_3d.models.tangent_vehicle_model import KinematicVehicleModel, VehicleModelConfig
from vehicle_3d.utils.ocp_util import OCPResults
from vehicle_3d.raceline.base_raceline import BaseRaceline, BasePlanarRaceline, RacelineConfig


class KinematicRaceline(BaseRaceline):
    ''' kinematic bicycle raceline '''
    model_config: VehicleModelConfig
    def __init__(self,
                 surf: BaseSurface,
                 config: RacelineConfig,
                 model_config: VehicleModelConfig,
                 ws_raceline: OCPResults = None,
                 ws_model: DynamicsModel = None):
        super().__init__(surf, config, model_config, ws_raceline, ws_model)

    def _get_model(self) -> KinematicVehicleModel:
        return KinematicVehicleModel(self.model_config, self.surf)

    def _get_state_bounds(self,n,k):
        state_u, state_l = super()._get_state_bounds(n,k)
        state_u[2] = np.pi
        state_l[2] =-np.pi
        state_l[3] = 0
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
        if self.config.ths0 is not None:
            g += [z0[2] - self.config.ths0]
            ubg += [0.]
            lbg += [0.]
        if self.config.v0 is not None:
            g += [z0[3] - self.config.v0]
            ubg += [0.]
            lbg += [0.]

    def _guess_z(self, n, k):
        z = super()._guess_z(n, k)
        z[3] = self.config.v_ws
        return z


class PlanarKinematicRaceline(KinematicRaceline, BasePlanarRaceline):
    ''' planar kinematic bicycle raceline '''
