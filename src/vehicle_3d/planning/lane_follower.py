'''
basic ocp planner for lane following
'''
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from vehicle_3d.pytypes import BaseBodyState
from vehicle_3d.utils.ocp_util import OCP, OCPConfig, OCPResults, Fix, OCPVars, \
    COLLOCATION_METHODS
from vehicle_3d.surfaces.base_surface import BaseCenterlineSurface
from vehicle_3d.models.dynamics_model import DynamicsModelConfig
from vehicle_3d.models.point_model import TangentPointModel
from vehicle_3d.models.tangent_vehicle_model import VehicleModelConfig, KinematicVehicleModel

@dataclass
class LaneFollowerConfig(OCPConfig):
    ''' configuration for lane follower solver '''
    mobile: bool = field(default = True)

    K: int = field(default = 3)
    N: int = field(default = 4)

    # interval between fixed space points
    ds: float = field(default = 5)

    # change default costs
    R: float = field(default = 0.1)
    dR: float = field(default = .1)

    y_cost: float = field(default = 1.0)
    ths_cost: float = field(default = 1.0)
    v_cost: float = field(default = 1.0)
    t_cost: float = field(default = 1.0)

    full_unpack: bool = field(default = False)

    y_sep_l: float = field(default = 0.0)
    '''tighten lane boundary to the left (y>0)'''
    y_sep_r: float = field(default = 0.0)
    '''tighten lane boundary to the right (y<0)'''

@dataclass
class LaneFollowerVars(OCPVars):
    ''' lane follower variables '''
    init_state: ca.SX = field(default = None)
    init_input: ca.SX = field(default = None)
    v_ref: ca.SX = field(default = None)
    ths_ref: ca.SX = field(default = None)
    y_ref: ca.SX = field(default = None)


class LaneFollower(OCP):
    ''' lane folling planner '''
    surf: BaseCenterlineSurface
    config: LaneFollowerConfig
    ocp_vars: LaneFollowerVars
    current_plan: OCPResults

    y_ref: float = 0.
    ''' y coordinate reference '''
    ths_ref: float = 0.
    ''' relative heading reference '''
    v_ref: float = 1.
    ''' speed reference '''

    def __init__(self,
            surf: BaseCenterlineSurface,
            config: LaneFollowerConfig,
            model_config: DynamicsModelConfig):
        config.closed = False
        super().__init__(surf, config, model_config, setup=True)

    def step(self, state: BaseBodyState):
        self.current_plan = super().step(state)

    def get_plan(self):
        ''' return the most recent plan '''
        return self.current_plan

    def _pre_setup_checks(self):
        super()._pre_setup_checks()
        if self.config.fix == Fix.S and not self.config.mobile:
            raise NotImplementedError('Online control fixed in space must be mobile')
        if self.config.fix == Fix.UNIFORM_DT:
            raise NotImplementedError('Online control with variable time must \
                be fixed in space')
        assert not self.config.closed

    def _initial_nlp_setup(self):
        self.ocp_vars = LaneFollowerVars()

    def _create_nlp_vars(self):
        super()._create_nlp_vars()
        self.ocp_vars.init_state = self.sym_class.sym('z0', self.ocp_vars.state_dim)
        self.ocp_vars.init_input = self.sym_class.sym('u0', self.ocp_vars.input_dim)
        self.ocp_vars.v_ref = self.sym_class.sym('v_ref')
        self.ocp_vars.ths_ref = self.sym_class.sym('ths_ref')
        self.ocp_vars.y_ref = self.sym_class.sym('y_ref')

    def _get_s(self, n, k):
        return self.config.ds * (n + self.config.tau[k])

    def _enforce_initial_constraints(self):
        super()._enforce_initial_constraints()

        Z = self.ocp_vars.Z
        U = self.ocp_vars.U
        init_state = self.ocp_vars.init_state
        init_input = self.ocp_vars.init_input

        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        if self.config.fix == Fix.S:
            g += [Z[0,0][1:] - init_state[1:]]
            ubg += [0.] * (self.ocp_vars.state_dim - 1)
            lbg += [0.] * (self.ocp_vars.state_dim - 1)
        else:
            g += [Z[0,0] - init_state]
            ubg += [0.] * self.ocp_vars.state_dim
            lbg += [0.] * self.ocp_vars.state_dim

        if self.config.method in COLLOCATION_METHODS:
            g += [U[0,0] - init_input]
            ubg += [0.] * self.ocp_vars.input_dim
            lbg += [0.] * self.ocp_vars.input_dim
        else:
            input_du, input_dl = self._get_input_rate_bounds(0,0)
            g += [(U[0,0] - init_input)*self.config.dt]
            ubg += input_du
            lbg += input_dl

    def _enforce_terminal_constraints(self):
        super()._enforce_terminal_constraints()

        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        g += [self._zF()[3] - self.ocp_vars.v_ref]
        ubg += [0.]
        lbg += [-np.inf]

    def _stage_cost(self, n, k):
        Q = ca.diag([self.config.y_cost, self.config.ths_cost, self.config.v_cost])
        e = self.ocp_vars.Z[n,k][1:4] \
            - ca.vertcat(self.ocp_vars.y_ref, self.ocp_vars.ths_ref, self.ocp_vars.v_ref)
        return ca.bilin(self.config.R, self.ocp_vars.U[n,k], self.ocp_vars.U[n,k]) + \
               ca.bilin(self.config.dR, self.ocp_vars.dU[n,k], self.ocp_vars.dU[n,k]) + \
               ca.bilin(Q, e, e) + \
               self.config.t_cost

    def _get_state_bounds(self, n, k):
        state_u, state_l = super()._get_state_bounds(n, k)

        state_u[1] -= self.config.y_sep_l
        state_l[1] += self.config.y_sep_r

        # modify some limits for forwards-only motion
        state_u[0] = np.inf
        state_l[0] =-np.inf
        if isinstance(self.model, TangentPointModel):
            state_l[2] = 0
        else:
            state_u[2] = np.pi
            state_l[2] =-np.pi
            state_l[3] = 0
        return state_u, state_l

    def _add_params(self):
        p = []
        p += [self.ocp_vars.init_state]
        p += [self.ocp_vars.init_input]

        p += [self.ocp_vars.y_ref]
        p += [self.ocp_vars.ths_ref]
        p += [self.ocp_vars.v_ref]

        if self.config.mobile:
            for n in range(self.config.N):
                for k in range(self.config.K + 1):
                    p += [self.ocp_vars.P[n,k]]

        self.ocp_vars.p = ca.vertcat(*p)

    def _get_params(self):
        z0, u0 = self.model.state2zu(self.current_state)
        if self.config.fix == Fix.S:
            z0[0] = 0

        p = [z0, u0, [self.y_ref], [self.ths_ref], [self.v_ref]]

        if self.config.mobile:
            # surface parameters
            if self.config.fix == Fix.S:
                # parameters are always fixed if fixed in space
                tau = self.config.tau
                s0 = np.arange(self.config.N) * self.config.ds + self.current_state.p.s
                param_s = np.concatenate((tau * self.config.ds)[None] + s0[None].T)
            else:
                param_s = self.v_ref * self.f_sol_t(self.sol) + self.current_state.p.s

            if self.surf.config.closed:
                param_s = np.mod(param_s, self.surf.s_max())

            param_terms = self.surf.sym_rep.f_param_terms(param_s.reshape((1,-1)), 0, 0)

            p_surf = np.concatenate(np.array(param_terms).T)

            p  += [p_surf]

        return np.concatenate(p)


class KinematicLaneFollower(LaneFollower):
    ''' kinematic bicycle lane follower '''
    model: KinematicVehicleModel
    def __init__(self,
            surf: BaseCenterlineSurface,
            config: LaneFollowerConfig,
            model_config: VehicleModelConfig):
        super().__init__(surf, config, model_config)

    def _get_model(self) -> KinematicVehicleModel:
        return KinematicVehicleModel(self.model_config, self.surf)
