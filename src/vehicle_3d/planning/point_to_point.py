'''
point to point planner
'''
from dataclasses import dataclass, field
from typing import List

import numpy as np
import casadi as ca

from vehicle_3d.pytypes import PythonMsg
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.tangent_vehicle_model import TangentVehicleModel, VehicleModelConfig, \
    KinematicVehicleModel
from vehicle_3d.utils.ocp_util import OCP, OCPConfig, Fix, COLLOCATION_METHODS, OCPResults, OCPVars
from vehicle_3d.obstacles.polytopes import RectangleObstacle

@dataclass
class PointPlannerConfig(OCPConfig):
    ''' point to point planner config '''
    N: int = field(default = 10)
    R: np.ndarray = field(default = .1)
    dR: np.ndarray = field(default = .1)
    v_ws: float = field(default = 1.)


class PointPlanner(OCP):
    ''' point to point planner, no obstacle avoidance'''
    solver_p: List[float]
    model: TangentVehicleModel
    model_config: VehicleModelConfig

    def __init__(self,
            surf: BaseSurface,
            config: PointPlannerConfig,
            model_config: VehicleModelConfig,
            ws_results: OCPResults = None,
            ws_model: TangentVehicleModel = None):
        assert config.method in COLLOCATION_METHODS
        config.fix = Fix.UNIFORM_DT
        config.closed = False
        config.mobile = False
        super().__init__(surf, config, model_config, ws_results, ws_model)

    def solve(self, init_state: PythonMsg, final_state: PythonMsg) -> OCPResults:
        # pylint: disable=arguments-differ
        # different arguments so that initial and final state can be changed easily

        z0, _ = self.model.state2zu(init_state)
        zf, _ = self.model.state2zu(final_state)
        self.solver_p = [*z0, *zf]
        return super().solve()

    def _create_nlp_vars(self):
        super()._create_nlp_vars()
        self.ocp_vars.p = []

    def _enforce_initial_constraints(self):
        p = self.ocp_vars.p
        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        init_state = self.sym_class.sym('z0', self.ocp_vars.state_dim)

        p += [init_state]
        g += [self.ocp_vars.Z[0,0] - init_state]
        ubg += [0.] * self.ocp_vars.state_dim
        lbg += [0.] * self.ocp_vars.state_dim

    def _enforce_terminal_constraints(self):
        p = self.ocp_vars.p
        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        final_state = self.sym_class.sym('zf', self.ocp_vars.state_dim)

        p += [final_state]
        g += [self._zF() - final_state]
        ubg += [0.] * self.ocp_vars.state_dim
        lbg += [0.] * self.ocp_vars.state_dim

    def _guess_z(self, n, k):
        z = [0.] * self.ocp_vars.state_dim
        z[3] = self.config.v_ws
        return z

    def _add_params(self):
        self.ocp_vars.p = ca.vertcat(*self.ocp_vars.p)

    def _get_params(self):
        return self.solver_p


class KinematicPointPlanner(PointPlanner):
    ''' point to point planner with kinematic bicycle model '''

    def _get_model(self) -> KinematicVehicleModel:
        return KinematicVehicleModel(self.model_config, self.surf)

@dataclass
class OBCAPointPlannerConfig(PointPlannerConfig):
    ''' obca point planner config '''
    d_min: float = field(default = 0.5)

@dataclass
class OBCAOCPVars(OCPVars):
    ''' obca point planner variables '''
    L: np.ndarray = field(default = None)
    M: np.ndarray = field(default = None)


class OBCAPointPlanner(PointPlanner):
    '''
    point to point planning with obstacles
    obstacle avoidance based on
    "Optimization-Based Collision Avoidance"
    available: https://arxiv.org/pdf/1711.03449.pdf
    '''
    obstacles: List[RectangleObstacle]
    ocp_vars: OBCAOCPVars
    def __init__(self,
            surf: BaseSurface,
            config: PointPlannerConfig,
            model_config: VehicleModelConfig,
            obstacles: List[RectangleObstacle],
            ws_results: OCPResults = None,
            ws_model: TangentVehicleModel = None):
        self.obstacles = obstacles
        super().__init__(surf, config, model_config, ws_results, ws_model)

    def _initial_nlp_setup(self):
        self.ocp_vars = OBCAOCPVars()

        if self.surf.config.mx_xp:
            self.sym_class = ca.MX

    def _create_nlp_vars(self):
        super()._create_nlp_vars()

        n_obs = len(self.obstacles)
        L = np.resize(np.array([], dtype = self.sym_class), (self.config.N, self.config.K+1, n_obs))
        M = np.resize(np.array([], dtype = self.sym_class), (self.config.N, self.config.K+1, n_obs))

        for n in range(self.config.N):
            for k in range(0, self.config.K+1):
                for m in range(n_obs):
                    if isinstance(self.obstacles[m], RectangleObstacle):
                        obs_b = self.obstacles[m].b
                        n_duals = obs_b.shape[0]

                        # add dual variables
                        L[n,k,m] = self.sym_class.sym(f'l_{n}_{k}_{m}', n_duals)
                        M[n,k,m] = self.sym_class.sym(f'm_{n}_{k}_{m}', n_duals)
                    else:
                        raise NotImplementedError(f'Unsupported obstacle class {self.obstacles[m]}')

        self.ocp_vars.L = L
        self.ocp_vars.M = M

    def _enforce_stage_constraints(self, n, k):
        super()._enforce_stage_constraints(n, k)

        Z = self.ocp_vars.Z
        L = self.ocp_vars.L
        M = self.ocp_vars.M

        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        veh_G, veh_g = self.model_config.get_polytope()

        veh_R = self.surf.pths2Rths(Z[n,k][0], Z[n,k][1], Z[n,k][2])
        veh_t = self.surf.p2x(Z[n,k][0], Z[n,k][1], self.model_config.h)

        n_obs = len(self.obstacles)

        for m in range(n_obs):
            if isinstance(self.obstacles[m], RectangleObstacle):
                obs_A = self.obstacles[m].A
                obs_b = self.obstacles[m].b
                n_duals = obs_b.shape[0]

                # add constraints
                c1 = ca.dot(-veh_g, M[n,k,m]) + (obs_A @ veh_t - obs_b).T @ L[n,k,m] # >= d_min
                c2 = veh_G.T @ M[n,k,m] + veh_R.T @ obs_A.T @ L[n,k,m]  # = [0,0,0]
                c3 = ca.dot(obs_A.T @ L[n,k,m], obs_A.T @ L[n,k,m]) # <= 1

                g +=  [c1,c2,c3]
                ubg += [np.inf]  + [0] * int(n_duals/2) + [1]
                lbg += [self.config.d_min] + [0] * int(n_duals/2) + [-np.inf]
            else:
                raise NotImplementedError(f'Unsupported obstacle class {self.obstacles[m]}')

    def _add_stage_decision_variables(self, n, k, w, w0, ubw, lbw):
        super()._add_stage_decision_variables(n, k, w, w0, ubw, lbw)

        L = self.ocp_vars.L
        M = self.ocp_vars.M

        n_obs = len(self.obstacles)

        for m in range(n_obs):
            obs_b = self.obstacles[m].b
            n_duals = obs_b.shape[0]

            w += [L[n,k,m]]
            w0 += [0.] * n_duals
            lbw += [0.] * n_duals
            ubw += [np.inf] * n_duals

            w += [M[n,k,m]]
            w0 += [0.] * n_duals
            lbw += [0.] * n_duals
            ubw += [np.inf] * n_duals


class KinematicOBCAPointPlanner(OBCAPointPlanner, KinematicPointPlanner):
    ''' point to point OBCA planner with kinematic bicycle model '''

    def _get_model(self) -> KinematicVehicleModel:
        return KinematicVehicleModel(self.model_config, self.surf)
