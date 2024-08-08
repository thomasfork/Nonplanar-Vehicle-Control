''' tangent vehicle model '''

from dataclasses import dataclass, field
from typing import List, Dict, Any

import casadi as ca
import numpy as np

from vehicle_3d.pytypes import BaseTangentBodyState
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.tire_model import TireFlags, TireInputChoices, get_tire
from vehicle_3d.models.vehicle_model import VehicleModel, DAEVehicleModel,\
    VehicleModelConfig, VehicleModelVars, DAEVehicleModelVars, BaseVehicleState

@dataclass
class TangentVehicleState(BaseVehicleState, BaseTangentBodyState):
    ''' tangent contact vehicle state '''

@dataclass
class TangentVehicleModelVars(VehicleModelVars):
    ''' variables for any tangent vehicle model '''
    ths: ca.SX = field(default = None)
    ths_dot: ca.SX = field(default = None)

    def pose_eqns(self, surf:BaseSurface, planar: bool = False):
        ''' compute w1 and w2 from velocity, and fill in '''
        surf_func = surf.pose_eqns_2D_planar if planar else surf.pose_eqns_2D
        self.s_dot, self.y_dot, self.ths_dot, self.w1, self.w2 = surf_func(
            self.v1, self.v2, self.w3, self.n
        )
        self.v3 = 0
        self.v3_dot = 0
        self.n_dot = 0
        self.vb = ca.vertcat(self.v1, self.v2, self.v3)
        self.wb = ca.vertcat(self.w1, self.w2, self.w3)

    def velocity_dynamics(self, config: VehicleModelConfig):
        ''' dynamics for v1, v2, w3 '''
        self.v1_dot = self.F[0] / config.m + self.w3*self.v2
        self.v2_dot = self.F[1] / config.m - self.w3*self.v1
        self.w3_dot = (self.K[2] - (config.I2 - config.I1) * self.w1*self.w2) / config.I3

    def approx_angular_velocity_rates(self, surf:BaseSurface, planar: bool = False):
        ''' approximate w1_dot and w2_dot from v1_dot and v2_dot '''
        if planar:
            self.w1_dot = 0
            self.w2_dot = 0
            self.v3_dot = 0
            return
        one = surf.sym_rep.one
        two = surf.sym_rep.two
        n = self.n
        v1_dot = self.v1_dot
        v2_dot = self.v2_dot
        J = surf.sym_rep.J
        J_inv = surf.sym_rep.J_inv
        dw1w2 = J_inv @ two @ ca.inv(one - n * two) @ J @ ca.vertcat(v1_dot,v2_dot)
        self.w2_dot = -dw1w2[0]
        self.w1_dot =  dw1w2[1]
        self.v3_dot = 0

    def approx_weight_distribution(self, config: VehicleModelConfig):
        ''' approximate weight distribution for all four tires '''
        F3_required = config.m * (self.v3_dot + self.v2*self.w1 - self.v1*self.w2)
        K1_required = config.I1 * self.w1_dot + \
            (config.I3 - config.I2)*self.w2*self.w3
        K2_required = config.I2 * self.w2_dot + \
            (config.I1 - config.I3)*self.w3*self.w1

        F3_N_required = F3_required - self.Fd[2] - self.Fg[2]
        K1_N_required = K1_required - self.Kd[0] - config.h * self.Ft[1]
        K2_N_required = K2_required - self.Kd[1] + config.h * self.Ft[0]

        Nf_required = config.lr / config.L * F3_N_required \
            - 1/config.L * K2_N_required
        Nr_required = config.lf / config.L * F3_N_required \
            + 1/config.L * K2_N_required
        Delta_required = 1/2/(config.tf**2 + config.tr**2) \
            * K1_N_required

        # tire normal forces
        Nfr = Nf_required/2 - config.tf * Delta_required
        Nfl = Nf_required/2 + config.tf * Delta_required
        Nrr = Nr_required/2 - config.tr * Delta_required
        Nrl = Nr_required/2 + config.tr * Delta_required
        return Nfr, Nfl, Nrr, Nrl


class TangentVehicleModel(VehicleModel):
    ''' generic tangent contact vehicle model class '''
    surf: BaseSurface
    model_vars: TangentVehicleModelVars
    config: VehicleModelConfig

    f_ths : ca.Function

    def _clear_model_vars(self):
        self.model_vars = TangentVehicleModelVars()

    def _add_pose_vars(self):
        self.model_vars.s = self.surf.sym_rep.s
        self.model_vars.y = self.surf.sym_rep.y
        self.model_vars.n = self.config.h
        self.model_vars.ths = self.surf.sym_rep.ths
        self.model_vars.R = self.surf.sym_rep.R_ths

        self.model_vars.p = ca.vertcat(
            self.model_vars.s,
            self.model_vars.y,
            self.model_vars.n
        )

    def _add_vel_vars(self):
        # override for kinematic models
        self.model_vars.v1 = ca.SX.sym('v1')
        self.model_vars.v2 = ca.SX.sym('v2')
        self.model_vars.w3 = ca.SX.sym('w3')
        self.model_vars.pose_eqns(self.surf, self.config.build_planar_model)

    def _add_two_track_tires(self, common_args: Dict[str, Any], iter_args: List[Dict[str, Any]]):
        uy = self.model_vars.uy
        yfr = ca.arctan(ca.tan(uy) / \
            ( self.config.tf/2/self.config.L * ca.tan(uy) + 1))
        yfl = ca.arctan(ca.tan(uy) / \
            (-self.config.tf/2/self.config.L * ca.tan(uy) + 1))
        yrr = 0
        yrl = 0
        tire_y = [yfr, yfl, yrr, yrl]
        tire_x = [
            ca.vertcat(self.config.lf, -self.config.tf, -self.config.h),
            ca.vertcat(self.config.lf, self.config.tf, -self.config.h),
            ca.vertcat(-self.config.lr, -self.config.tr, -self.config.h),
            ca.vertcat(-self.config.lr, self.config.tr, -self.config.h),
        ]
        self.tires = []
        for x, y, iter_arg in zip(tire_x, tire_y, iter_args):
            self.tires.append(get_tire(
                x = x,
                y = y,
                **iter_arg,
                **common_args
            ))

    def _state_derivative(self):
        # override for kinematic models, extend for some dynamic models
        self.model_vars.velocity_dynamics(self.config)
        self.model_vars.approx_angular_velocity_rates(self.surf, self.config.build_planar_model)

        self.model_vars.z = ca.vertcat(
            self.model_vars.s,
            self.model_vars.y,
            self.model_vars.ths,
            self.model_vars.v1,
            self.model_vars.v2,
            self.model_vars.w3
        )
        self.model_vars.z_dot = ca.vertcat(
            self.model_vars.s_dot,
            self.model_vars.y_dot,
            self.model_vars.ths_dot,
            self.model_vars.v1_dot,
            self.model_vars.v2_dot,
            self.model_vars.w3_dot
        )

    def _setup_helper_functions(self):
        super()._setup_helper_functions()
        self.f_ths = self.surf.fill_in_param_terms(
            [self.model_vars.ths],
            [self.model_vars.z]
        )

    def get_empty_state(self) -> TangentVehicleState:
        state = TangentVehicleState()
        state.p.n = self.config.h
        state.ab.a3 = self.config.g
        state.tfr.N = self.config.lr / self.config.L * self.config.m * self.config.g / 2
        state.tfl.N = self.config.lr / self.config.L * self.config.m * self.config.g / 2
        state.trr.N = self.config.lf / self.config.L * self.config.m * self.config.g / 2
        state.trl.N = self.config.lf / self.config.L * self.config.m * self.config.g / 2
        self.surf.p2gx(state)
        return state

    def state2z(self, state: TangentVehicleState):
        return [state.p.s, state.p.y, state.ths, state.vb.v1, state.vb.v2, state.wb.w3]

    def zu2state(self, state: TangentVehicleState, z, u):
        super().zu2state(state, z, u)
        state.ths = float(self.f_ths(z))

    def zu(self, s: float = 0):
        return [self.surf.s_max(s), self.surf.y_max(s), \
                np.inf, np.inf, np.inf, np.inf]

    def zl(self, s: float = 0):
        return [self.surf.s_min(), self.surf.y_min(s), \
                -np.inf, -np.inf, -np.inf, -np.inf]


class TangentDAEVehicleModelVars(TangentVehicleModelVars, DAEVehicleModelVars):
    ''' variables for any tangent DAE vehicle model '''


class TangentDAEVehicleModel(TangentVehicleModel, DAEVehicleModel):
    ''' generic tangent contact DAE vehicle model class '''
    model_vars: TangentDAEVehicleModelVars

    def _clear_model_vars(self):
        self.model_vars = TangentDAEVehicleModelVars()

    def zua2state(self, state: TangentVehicleState, z, u, a):
        super().zua2state(state, z, u, a)
        state.ths = float(self.f_ths(z))


@dataclass
class KinematicVehicleModelVars(TangentVehicleModelVars):
    ''' kinematic vehicle model variables '''
    beta: ca.SX = field(default = None)
    v: ca.SX = field(default = None)
    v_dot: ca.SX = field(default = None)


class KinematicVehicleModel(TangentVehicleModel):
    ''' kinematic vehicle model'''
    model_vars: KinematicVehicleModelVars

    def _clear_model_vars(self):
        self.model_vars = KinematicVehicleModelVars()

    def _add_vel_vars(self):
        # pylint: disable=attribute-defined-outside-init
        self.model_vars.v = ca.SX.sym('v')
        uy = self.model_vars.uy
        self.model_vars.beta = ca.atan(self.config.lr / (self.config.L) * ca.tan(uy))
        self.model_vars.v1 = ca.cos(self.model_vars.beta) * self.model_vars.v
        self.model_vars.v2 = ca.sin(self.model_vars.beta) * self.model_vars.v
        self.model_vars.w3 = self.model_vars.v * ca.cos(self.model_vars.beta) \
            / (self.config.L) * ca.tan(uy)

        self.model_vars.pose_eqns(self.surf, self.config.build_planar_model)

    def _tire_forces(self):
        # zeroed and estimated later
        self.model_vars.Ft = ca.vertcat(0., 0., 0.)
        self.model_vars.Kt = ca.vertcat(0., 0., 0.)

    def _state_derivative(self):
        # pylint: disable=attribute-defined-outside-init
        F1 = self.model_vars.F[0]
        F2 = self.model_vars.F[1]
        beta = self.model_vars.beta
        a = (F1 * ca.cos(beta) + F2 * ca.sin(beta)) / self.config.m
        self.model_vars.v_dot = self.model_vars.ua + a - self.model_vars.ub

        self.model_vars.z = ca.vertcat(
            self.model_vars.s,
            self.model_vars.y,
            self.model_vars.ths,
            self.model_vars.v
        )
        self.model_vars.z_dot = ca.vertcat(
            self.model_vars.s_dot,
            self.model_vars.y_dot,
            self.model_vars.ths_dot,
            self.model_vars.v_dot
        )

        # approximate velocity change
        self.model_vars.v1_dot = self.model_vars.v_dot * ca.cos(beta)
        self.model_vars.v2_dot = self.model_vars.v_dot * ca.sin(beta)
        uy = self.model_vars.uy
        self.model_vars.w3_dot = self.model_vars.v_dot * ca.cos(beta) / (self.config.L) \
            * ca.tan(uy)
        self.model_vars.approx_angular_velocity_rates(self.surf, self.config.build_planar_model)

    def _calc_outputs(self):
        ua = self.model_vars.ua
        uy = self.model_vars.uy
        ub = self.model_vars.ub
        v = self.model_vars.v
        beta = self.model_vars.beta
        Fg1 = self.model_vars.Fg[0]
        Fg2 = self.model_vars.Fg[1]
        agt = -(-Fg1 * ca.sin(beta) + Fg2 * ca.cos(beta)) / self.config.m

        # approximate tire forces
        Ftl = (ua - ub) * self.config.m
        Ftt = (agt + v**2 * uy / self.config.L) * self.config.m
        Ft1 = ca.cos(beta) * Ftl - ca.sin(beta) * Ftt
        Ft2 = ca.sin(beta) * Ftl + ca.cos(beta) * Ftt
        self.model_vars.Ft = ca.vertcat(Ft1, Ft2, 0)
        self.model_vars.N = ca.vertcat(
            *self.model_vars.approx_weight_distribution(self.config)
        )
        self.model_vars.N_reg = self.model_vars.N
        self.model_vars.Ft = ca.vertcat(Ft1, Ft2, sum(self.model_vars.N[k] for k in range(4)))

        # approximate body acceleration
        self.model_vars.F = self.model_vars.Fg + self.model_vars.Fd + self.model_vars.Ft
        self.model_vars.K = self.model_vars.Kd + self.model_vars.Kt
        self.model_vars.ab = (self.model_vars.F - self.model_vars.Fg) / self.config.m

        # set up friction cone limit
        N_tot = sum(self.model_vars.N[k] for k in range(4))
        allowed_accel = N_tot * self.config.tire_config.mu / self.config.m
        used_accel_sq = (self.model_vars.v**2 * uy / self.config.L + agt)**2 + (ua - ub)**2

        self.model_vars.g = ca.vertcat(
            self.model_vars.N / self.config.force_normalization_const(),
            used_accel_sq / allowed_accel**2
        )
        self.model_vars.ubg = [
            self.config.N_max / self.config.force_normalization_const(),
            self.config.N_max / self.config.force_normalization_const(),
            self.config.N_max / self.config.force_normalization_const(),
            self.config.N_max / self.config.force_normalization_const(),
            1.0
        ]
        self.model_vars.lbg = [
            0., 0., 0., 0.,
            -np.inf
        ]

        # add visual tires
        common_tire_args = {
            'config': self.config.tire_config,
            'vb': self.model_vars.vb,
            'wb': self.model_vars.wb,
            'mu': self.config.tire_config.mu,
            'input_mode': TireInputChoices.FREE_SPIN,
            'flags': [TireFlags.NO_FORCES],
            'visual_force_norm': self.config.m * self.config.g / self.config.L,
        }
        iter_tire_args = [{'N':self.model_vars.N[k]} for k in range(4)]
        self._add_two_track_tires(common_tire_args, iter_tire_args)

    def state2z(self, state: TangentVehicleState):
        return [state.p.s, state.p.y, state.ths, state.vb.signed_mag()]

    def zu(self, s: float = 0):
        return [self.surf.s_max(s), self.surf.y_max(s), np.inf, np.inf]

    def zl(self, s: float = 0):
        return [self.surf.s_min(), self.surf.y_min(s), -np.inf, -np.inf]

    def get_color(self) -> List[float]:
        if self.config.build_planar_model:
            return [0.5, 0.7, 0.8, 1.0]
        return [0.2, 0.2, 1.0, 1.0]

    def get_label(self):
        if self.config.build_planar_model:
            return 'Planar Kin. Bicycle'
        return 'Kinematic Bicycle'


class DynamicTwoTrackSlipInputVehicleModel(TangentDAEVehicleModel):
    ''' two track longitudinal slip input model '''

    def _tire_forces(self):
        Nfr = ca.SX.sym('Nfr')
        Nfl = ca.SX.sym('Nfl')
        Nrr = ca.SX.sym('Nrr')
        Nrl = ca.SX.sym('Nrl')
        self.model_vars.N = ca.vertcat(Nfr, Nfl, Nrr, Nrl)
        self.model_vars.N_reg = self.model_vars.N * self.config.force_normalization_const()

        common_tire_args = {
            'config': self.config.tire_config,
            'vb': self.model_vars.vb,
            'wb': self.model_vars.wb,
            'mu': self.config.tire_config.mu,
            'input_mode': TireInputChoices.LONGITUDINAL_SLIP,
            'visual_force_norm': self.config.m * self.config.g / self.config.L,
        }
        iter_tire_args = [{'N':self.model_vars.N_reg[k]} for k in range(4)]
        self._add_two_track_tires(common_tire_args, iter_tire_args)

        self.model_vars.u = ca.vertcat(
            *[tire.s for tire in self.tires],
            self.model_vars.uy
        )

        self.model_vars.Ft = sum(tire.Fb for tire in self.tires)
        self.model_vars.Kt = sum(tire.Kb for tire in self.tires)

    def _state_derivative(self):
        super()._state_derivative()
        # pylint: disable=attribute-defined-outside-init
        N_eq = self.model_vars.approx_weight_distribution(self.config)
        self.model_vars.a = self.model_vars.N
        self.model_vars.h = self.model_vars.N \
            - ca.vertcat(*N_eq) / self.config.force_normalization_const()

    def _calc_outputs(self):
        self.model_vars.g = []
        self.model_vars.ubg = []
        self.model_vars.lbg = []

    def state2u(self, state: TangentVehicleState):
        return [state.tfr.s,
                state.tfl.s,
                state.trr.s,
                state.trl.s,
                state.u.y]

    def state2a(self, state: TangentVehicleState):
        return [state.tfr.N / self.config.force_normalization_const(),
                state.tfl.N / self.config.force_normalization_const(),
                state.trr.N / self.config.force_normalization_const(),
                state.trl.N / self.config.force_normalization_const()]

    def u2state(self, state: TangentVehicleState, u):
        u = self._coerce_input_limits(u)
        state.tfr.s = float(u[0])
        state.tfl.s = float(u[1])
        state.trr.s = float(u[2])
        state.trl.s = float(u[3])
        state.u.y       = float(u[4])

    def uu(self):
        return [1.0] * 4 + [self.config.uy_max]

    def ul(self):
        return [-1.0] * 4 + [self.config.uy_min]

    def duu(self):
        return [1.0] * 4 + [self.config.duy_max]

    def dul(self):
        return [-1.0] * 4 + [self.config.duy_min]

    def au(self):
        return [self.config.N_max / self.config.force_normalization_const()] * 4

    def al(self):
        return [0.] * 4

    def get_color(self) -> List[float]:
        if self.config.build_planar_model:
            return [1.0, 0.7, 0.8, 1.0]
        return [1.0, 0.2, 0.2, 1.0]

    def get_label(self):
        if self.config.build_planar_model:
            return 'Planar Two Track Slip Input'
        return 'Two Track Slip Input'


class DynamicTwoTrackTorqueInputVehicleModel(TangentDAEVehicleModel):
    ''' two track wheel torque input model '''
    model_vars: TangentDAEVehicleModelVars

    def _tire_forces(self):
        Nfr = ca.SX.sym('Nfr')
        Nfl = ca.SX.sym('Nfl')
        Nrr = ca.SX.sym('Nrr')
        Nrl = ca.SX.sym('Nrl')
        self.model_vars.N = ca.vertcat(Nfr, Nfl, Nrr, Nrl)
        self.model_vars.N_reg = self.model_vars.N * self.config.force_normalization_const()

        common_tire_args = {
            'config': self.config.tire_config,
            'vb': self.model_vars.vb,
            'wb': self.model_vars.wb,
            'mu': self.config.tire_config.mu,
            'input_mode': TireInputChoices.WHEEL_TORQUE,
            'visual_force_norm': self.config.m * self.config.g / self.config.L,
        }
        iter_tire_args = [{'N':self.model_vars.N_reg[k]} for k in range(4)]
        self._add_two_track_tires(common_tire_args, iter_tire_args)

        self.model_vars.u = ca.vertcat(
            *[tire.T for tire in self.tires],
            self.model_vars.uy
        )

        self.model_vars.Ft = sum(tire.Fb for tire in self.tires)
        self.model_vars.Kt = sum(tire.Kb for tire in self.tires)

    def _state_derivative(self):
        super()._state_derivative()
        # pylint: disable=attribute-defined-outside-init
        N_eq = self.model_vars.approx_weight_distribution(self.config)
        self.model_vars.a = self.model_vars.N
        self.model_vars.h = self.model_vars.N \
            - ca.vertcat(*N_eq) / self.config.force_normalization_const()

        self.model_vars.z = ca.vertcat(
            self.model_vars.z,
            self.tires[0].w,
            self.tires[1].w,
            self.tires[2].w,
            self.tires[3].w,
        )
        self.model_vars.z_dot = ca.vertcat(
            self.model_vars.z_dot,
            self.tires[0].w_dot,
            self.tires[1].w_dot,
            self.tires[2].w_dot,
            self.tires[3].w_dot,
        )

    def _calc_outputs(self):
        self.model_vars.g = []
        self.model_vars.ubg = []
        self.model_vars.lbg = []

    def state2z(self, state: TangentVehicleState):
        return super().state2z(state) + [
            state.tfr.w,
            state.tfl.w,
            state.trr.w,
            state.trl.w,
        ]

    def state2u(self, state: TangentVehicleState):
        return [state.tfr.T,
                state.tfl.T,
                state.trr.T,
                state.trl.T,
                state.u.y]

    def state2a(self, state: TangentVehicleState):
        return [state.tfr.N / self.config.force_normalization_const(),
                state.tfl.N / self.config.force_normalization_const(),
                state.trr.N / self.config.force_normalization_const(),
                state.trl.N / self.config.force_normalization_const()]

    def u2state(self, state: TangentVehicleState, u):
        u = self._coerce_input_limits(u)
        state.tfr.T = float(u[0])
        state.tfl.T = float(u[1])
        state.trr.T = float(u[2])
        state.trl.T = float(u[3])
        state.u.y       = float(u[4])

    def zu(self, s = 0):
        return super().zu(s) + [np.inf]*4

    def zl(self, s = 0):
        return super().zl(s) + [-np.inf]*4

    def uu(self):
        return [np.inf] * 4 + [self.config.uy_max]

    def ul(self):
        return [-np.inf] * 4 + [self.config.uy_min]

    def duu(self):
        return [np.inf] * 4 + [self.config.duy_max]

    def dul(self):
        return [-np.inf] * 4 + [self.config.duy_min]

    def au(self):
        return [self.config.N_max / self.config.force_normalization_const()] * 4

    def al(self):
        return [0.] * 4

    def get_color(self) -> List[float]:
        if self.config.build_planar_model:
            return [1.0, 0.7, 0.8, 1.0]
        return [1.0, 0.2, 0.2, 1.0]

    def get_label(self):
        if self.config.build_planar_model:
            return 'Planar Two Track Torque Input'
        return 'Two Track Torque Input'
