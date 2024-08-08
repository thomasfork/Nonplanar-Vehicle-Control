'''
vehicle models that are not constrained to remain tangent to a surface
work in progress!
'''

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from vehicle_3d.pytypes import RelativeOrientation
from vehicle_3d.utils.rotations import Rotation, Reference, Parameterization, RelativeEulerAngles
from vehicle_3d.utils.ca_utils import hat, ca_smooth_geq
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.vehicle_model import VehicleModel, \
    VehicleModelConfig, VehicleModelVars, BaseVehicleState
from vehicle_3d.models.tire_model import TireInputChoices, get_tire

@dataclass
class FreeVehicleState(BaseVehicleState):
    ''' free vehicle state '''
    r: RelativeOrientation = field(default = None)

    def __post_init__(self):
        if self.r is None:
            self.r = RelativeEulerAngles()
        super().__post_init__()

    @property
    def ths(self):
        ''' relative heading angle estimate '''
        return self.r.ths

    @ths.setter
    def _ths_setter(self, val):
        self.r.from_yaw(val)

@dataclass
class FreeVehicleModelVars(VehicleModelVars):
    ''' variables for any free motion vehicle model '''
    r: ca.SX = field(default = None)
    r_dot: ca.SX = field(default = None)
    p_dot: ca.SX = field(default = None)
    w_rel: ca.SX = field(default = None)

@dataclass
class FreeVehicleModelConfig(VehicleModelConfig):
    ''' configuration for free motion vehicle model '''
    zeta: float = field(default = 0.7)
    ''' suspension damping ratio '''
    wn:   float = field(default = 6.0)
    ''' suspension resonant frequency '''
    q: float = field(default = 1.0)
    ''' suspension stiffness scaling coefficient '''


class FreeSlipInputVehicleModel(VehicleModel):
    '''
    slip input four-wheeled vehicle with suspension model
    Vehicle is assumed to remain nearly upright but can move up/down and roll/pitch
      spring-damper forces at all four corners are added for a simplified suspension,
    which keeps the vehicle from falling through the surface and causes roll/pitch motion
    when accelerating
      This vehicle model can fly off of the surface it is on,
    but will be subject to pose eqn. constraints.

    main simplifications:
        tire forces act at a fixed location in the vehicle body frame
        tire forces act in fixed directions in the vehicle body frame
            ex: normal force direction on tire changes at large roll angles.
        tire camber angle is not imparted to the tire model
    '''
    surf: BaseSurface
    rot: Rotation
    model_vars: FreeVehicleModelVars
    config: FreeVehicleModelConfig

    # internal helper functions for tire camber and steering angles
    # not currently used but may be used to recalculate both, c.f. motorcycle model
    _f_tire_c: ca.Function
    _f_tire_y: ca.Function

    f_r : ca.Function

    def __init__(self, config: FreeVehicleModelConfig, surf: BaseSurface):
        self.rot = Rotation(
            ref = Reference.PARAMETRIC,
            param = Parameterization.YPR
        )
        super().__init__(config, surf)

    def _clear_model_vars(self):
        self.model_vars = FreeVehicleModelVars()

    def _add_pose_vars(self):
        self.model_vars.s = self.surf.sym_rep.s
        self.model_vars.y = self.surf.sym_rep.y
        self.model_vars.n = self.surf.sym_rep.n
        self.model_vars.p = self.surf.sym_rep.p
        self.model_vars.r = self.rot.r()
        if self.rot.ref == Reference.PARAMETRIC:
            self.model_vars.R = self.surf.sym_rep.Rp @ self.rot.R()
            R_rel = self.rot.R()
        else:
            self.model_vars.R = self.rot.R()
            R_rel = self.surf.sym_rep.Rp.T @ self.rot.R()

        uy = self.model_vars.uy
        R_u = ca.horzcat(
            ca.vertcat(ca.cos(uy), ca.sin(uy), 0),
            ca.vertcat(-ca.sin(uy),ca.cos(uy),0),
            ca.vertcat(0,0,1)
        )
        R_u_rel = R_rel @ R_u

        tire_c = -ca.arcsin(R_u_rel[2,1])
        tire_y = ca.arctan2(-R_u_rel[0,1],  R_u_rel[1,1]) - self.rot.yaw_angle()
        self._f_tire_c = self.surf.fill_in_param_terms(
            tire_c,
            [self.model_vars.p, self.model_vars.r, self.model_vars.uy])
        self._f_tire_y = self.surf.fill_in_param_terms(
            tire_y,
            [self.model_vars.p, self.model_vars.r, self.model_vars.uy])

    def _add_vel_vars(self):
        vb = ca.SX.sym('vb', 3)
        wb = ca.SX.sym('wb', 3)
        self.model_vars.v1 = vb[0]
        self.model_vars.v2 = vb[1]
        self.model_vars.v3 = vb[2]
        self.model_vars.vb = vb
        self.model_vars.w1 = wb[0]
        self.model_vars.w2 = wb[1]
        self.model_vars.w3 = wb[2]
        self.model_vars.wb = wb

        p_dot, r_dot, w_rel = self.surf.pose_eqns_3D(vb, wb, self.rot)

        self.model_vars.s_dot = p_dot[0]
        self.model_vars.y_dot = p_dot[1]
        self.model_vars.n_dot = p_dot[2]
        self.model_vars.p_dot = p_dot
        self.model_vars.r_dot = r_dot
        self.model_vars.w_rel = w_rel

    def _tire_normal_forces(self):
        ''' tire normal forces from mass spring damper at all four corners '''
        if self.rot.ref == Reference.GLOBAL:
            R_rel = self.surf.sym_rep.Rp.T @ self.rot.R()
        else:
            R_rel = self.rot.R()
        corner_xb = [
            ca.vertcat( self.config.lf, -self.config.tf, -self.config.h),
            ca.vertcat( self.config.lf,  self.config.tf, -self.config.h),
            ca.vertcat(-self.config.lr, -self.config.tr, -self.config.h),
            ca.vertcat(-self.config.lr,  self.config.tr, -self.config.h),
        ]

        k = self.config.m * self.config.wn**2 * self.config.q
        front_k = (self.config.lr / self.config.L) * k
        rear_k  = (self.config.lf / self.config.L) * k
        corner_k = [front_k, front_k, rear_k, rear_k]

        b = 2 * self.config.m * self.config.wn * self.config.zeta * self.config.q
        front_b = (self.config.lr / self.config.L) * b
        rear_b  = (self.config.lf / self.config.L) * b
        corner_b = [front_b, front_b, rear_b, rear_b]

        d = self.config.m * self.config.g / sum(corner_k)

        N = []
        for xb, k, b in zip(corner_xb, corner_k, corner_b):
            x_rel = ca.vertcat(0,0,self.model_vars.n) \
                + R_rel @ xb
            axial_displacement = x_rel[2] / R_rel[2,2] - d

            axial_displacement_rate = \
                ca.jacobian(axial_displacement, self.model_vars.p) @ self.model_vars.p_dot \
                + ca.jacobian(axial_displacement, self.model_vars.r) @ self.model_vars.r_dot
            N += [
                ca.fmax(0,
                    (-k * axial_displacement - b * axial_displacement_rate) \
                        * ca_smooth_geq(-axial_displacement)
                )
                ]
        self.model_vars.N = ca.vertcat(*N)
        self.model_vars.N_reg = self.model_vars.N

    def _tire_forces(self):
        self._tire_normal_forces()

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
        iter_args = [{'N':self.model_vars.N[k]} for k in range(4)]
        common_args = {
            'config': self.config.tire_config,
            'vb': self.model_vars.vb,
            'wb': self.model_vars.wb,
            'mu': self.config.tire_config.mu,
            'input_mode': TireInputChoices.LONGITUDINAL_SLIP,
            'visual_force_norm': self.config.m * self.config.g / self.config.L,
        }

        self.tires = []
        for x, y, iter_arg in zip(tire_x, tire_y, iter_args):
            c = self._f_tire_c(self.model_vars.p, self.model_vars.r, y)
            self.tires.append(get_tire(
                x = x,
                y = self._f_tire_y(self.model_vars.p, self.model_vars.r, y),
                **iter_arg,
                **common_args
            ))
            self.tires[-1].c = c

        self.model_vars.u = ca.vertcat(
            *[tire.s for tire in self.tires],
            self.model_vars.uy
        )

        self.model_vars.Ft = sum(tire.Fb for tire in self.tires)
        self.model_vars.Kt = sum(tire.Kb for tire in self.tires)

    def _state_derivative(self):
        vb = self.model_vars.vb
        wb = self.model_vars.wb
        Fb = self.model_vars.F
        Kb = self.model_vars.K
        Wb = hat(wb)
        Ib = np.diag([self.config.I1, self.config.I2, self.config.I3])

        vb_dot = Fb / self.config.m - Wb @ vb
        wb_dot = ca.inv(Ib) @ (Kb - Wb @ Ib @ wb)

        self.model_vars.v1_dot = vb_dot[0]
        self.model_vars.v2_dot = vb_dot[1]
        self.model_vars.v3_dot = vb_dot[2]
        self.model_vars.w1_dot = wb_dot[0]
        self.model_vars.w2_dot = wb_dot[1]
        self.model_vars.w3_dot = wb_dot[2]

        self.model_vars.z = ca.vertcat(
            self.model_vars.p,
            self.model_vars.r,
            vb,
            wb)
        self.model_vars.z_dot = ca.vertcat(
            self.model_vars.p_dot,
            self.model_vars.r_dot,
            vb_dot,
            wb_dot)

        self.model_vars.ab = (Fb - self.model_vars.Fg) / self.config.m

    def _calc_outputs(self):
        self.model_vars.g = []
        self.model_vars.ubg = []
        self.model_vars.lbg = []

    def _setup_helper_functions(self):
        super()._setup_helper_functions()
        self.f_r = self.surf.fill_in_param_terms(
            self.model_vars.r,
            self.model_vars.get_all_indep_vars()
        )

    def get_empty_state(self) -> FreeVehicleState:
        state = FreeVehicleState(r = self.rot.get_empty_state())
        state.p.n = self.config.h
        state.ab.a3 = self.config.g
        state.tfr.N = self.config.lr / self.config.L * self.config.m * self.config.g / 2
        state.tfl.N = self.config.lr / self.config.L * self.config.m * self.config.g / 2
        state.trr.N = self.config.lf / self.config.L * self.config.m * self.config.g / 2
        state.trl.N = self.config.lf / self.config.L * self.config.m * self.config.g / 2
        self.surf.p2gx(state)
        return state

    def state2u(self, state: FreeVehicleState):
        return [state.tfr.s,
                state.tfl.s,
                state.trr.s,
                state.trl.s,
                state.u.y]

    def state2z(self, state:FreeVehicleState):
        return [
            *state.p.to_vec(),
            *state.r.to_vec(),
            *state.vb.to_vec(),
            *state.wb.to_vec()
        ]

    def u2state(self, state: FreeVehicleState, u):
        u = self._coerce_input_limits(u)
        state.tfr.s = float(u[0])
        state.tfl.s = float(u[1])
        state.trr.s = float(u[2])
        state.trl.s = float(u[3])
        state.u.y   = float(u[4])

    def zu2state(self, state: FreeVehicleState, z, u):
        super().zu2state(state, z, u)
        state.r.from_vec(self.f_r(z,u))

    def zu(self, s: float = 0):
        return [
            self.surf.s_max(),
            self.surf.y_max(s),
            self.config.h * 1.1, #TODO
            *self.rot.ubr()] + \
            [np.inf] * 6

    def zl(self, s: float = 0):
        return [
            self.surf.s_min(),
            self.surf.y_min(s),
            self.config.h * 0.9, #TODO
            *self.rot.lbr()]  + \
            [-np.inf] * 6

    def uu(self):
        return [1.0] * 4 + [self.config.uy_max]

    def ul(self):
        return [-1.0] * 4 + [self.config.uy_min]

    def duu(self):
        return [1.0] * 4 + [self.config.duy_max]

    def dul(self):
        return [-1.0] * 4 + [self.config.duy_min]

    def get_color(self):
        if self.config.build_planar_model:
            return [0.0, 0.0, 0.0, 1.0]
        return [0.1, 0.1, 0.1, 1.0]

    def get_label(self):
        if self.config.build_planar_model:
            return 'Planar Suspended Slip Input'
        return 'Suspended Slip Input'


class FreeVehicleModel(FreeSlipInputVehicleModel):
    '''
    free vehicle model with torque input
    '''
    def _tire_forces(self):
        self._tire_normal_forces()

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
        iter_args = [{'N':self.model_vars.N[k]} for k in range(4)]
        common_args = {
            'config': self.config.tire_config,
            'vb': self.model_vars.vb,
            'wb': self.model_vars.wb,
            'mu': self.config.tire_config.mu,
            'input_mode': TireInputChoices.WHEEL_TORQUE,
            'visual_force_norm': self.config.m * self.config.g / self.config.L,
        }

        self.tires = []
        for x, y, iter_arg in zip(tire_x, tire_y, iter_args):
            c = self._f_tire_c(self.model_vars.p, self.model_vars.r, y)
            self.tires.append(get_tire(
                x = x,
                y = self._f_tire_y(self.model_vars.p, self.model_vars.r, y),
                **iter_arg,
                **common_args
            ))
            self.tires[-1].c = c

        self.model_vars.u = ca.vertcat(
            *[tire.T for tire in self.tires],
            self.model_vars.uy
        )

        self.model_vars.Ft = sum(tire.Fb for tire in self.tires)
        self.model_vars.Kt = sum(tire.Kb for tire in self.tires)

    def _state_derivative(self):
        super()._state_derivative()
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

    def state2u(self, state: FreeVehicleState):
        return [state.tfr.T,
                state.tfl.T,
                state.trr.T,
                state.trl.T,
                state.u.y]

    def u2state(self, state: FreeVehicleState, u):
        u = self._coerce_input_limits(u)
        state.tfr.T = float(u[0])
        state.tfl.T = float(u[1])
        state.trr.T = float(u[2])
        state.trl.T = float(u[3])
        state.u.y   = float(u[4])

    def state2z(self, state:FreeVehicleState):
        return super().state2z(state) + [
            state.tfr.w,
            state.tfl.w,
            state.trr.w,
            state.trl.w,
        ]

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
