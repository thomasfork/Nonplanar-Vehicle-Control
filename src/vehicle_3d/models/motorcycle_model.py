'''
geometric model of motorcycle chassis
state and input forces are normalized by m*g when vectorized for model
as this helps optimization
(structures contain MKS values)
'''

from typing import Dict, List
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from vehicle_3d.pytypes import VectorizablePythonMsg, BodyForce, BodyMoment, \
    BaseBodyState, NestedPythonMsg
from vehicle_3d.models.dynamics_model import DAEDynamicsModel, DynamicsModelConfig, \
    DAEDynamicsModelVars
from vehicle_3d.models.motorcycle_tire_model import MotorcycleTireConfig, \
    MotorcycleTireModel, MotorcycleTire, MotorcycleTireState
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.utils.ca_utils import hat, ca_abs
from vehicle_3d.visualization.utils import get_unit_arrow, join_vis, get_sphere
from vehicle_3d.visualization.objects import InstancedVertexObject, VertexObject, UBOObject
from vehicle_3d.visualization.gltf2 import load_motorcycle, load_split_motorcycle
from vehicle_3d.visualization.ca_glm import rotate, translate, translate_rel

@dataclass
class MotorcycleActuation(VectorizablePythonMsg):
    ''' general purpose high-level vehicle actuation '''
    Ff: float = field(default = 0.)
    ''' front tire force '''
    Fr: float = field(default = 0.)
    ''' rear tire force '''
    y: float = field(default = 0.)
    ''' front steering angle command '''
    d_ddot: float = field(default = 0.)
    ''' driver tilt input accel '''

    def to_vec(self):
        return np.array([self.Ff, self.Fr, self.y, self.d_ddot])

    def from_vec(self, vec: np.ndarray):
        self.Ff, self.Fr, self.y, self.d_ddot = vec

@dataclass
class BaseMotorcycleState(BaseBodyState, NestedPythonMsg):
    ''' motorcycle state '''
    u: MotorcycleActuation = field(default = None)
    F: BodyForce = field(default = None)
    K: BodyMoment = field(default = None)
    ths: float = field(default = 0.)
    d: float = field(default = 0.)
    ''' driver tilt input '''
    d_dot: float = field(default = 0.)
    ''' driver tilt input rate '''
    c: float = field(default = 0.)
    ''' camber angle '''
    c_dot: float = field(default = 0.)
    ''' camber angle rate '''
    c_ddot: float = field(default = 0.)
    v1_dot: float = field(default = 0.)
    v2_dot: float = field(default = 0.)
    w3_dot: float = field(default = 0.)

    tf: MotorcycleTireState = field(default = None)
    tr: MotorcycleTireState = field(default = None)

@dataclass
class MotorcycleConfig(DynamicsModelConfig, NestedPythonMsg):
    ''' configuration of motorcycle chassis and input limits '''
    m: float = field(default = 240)
    I1: float = field(default = 18)
    I2: float = field(default = 60)
    I3: float = field(default = 48)

    lf: float = field(default = 0.75)
    lr: float = field(default = 0.75)
    h: float = field(default = .5)
    head_angle: float = field(default = np.deg2rad(30))
    rake: float = field(default = 0.04)
    tire_rad: float = field(default = 0.1)
    tire_config: MotorcycleTireConfig = field(default = None)

    c_max: float = field(default = 1.5)
    ''' max camber angle in radians '''
    c_min: float = field(default = -1.5)
    ''' min camber angle in radians '''
    d_max: float = field(default = 0.05)
    ''' max rider displacement in meters '''
    d_min: float = field(default = -0.05)
    ''' min rider displacement in meters '''

    uy_max: float = field(default = 0.7)
    ''' max steering angle in radians '''
    uy_min: float = field(default =-0.7)
    ''' min steering angle in radians '''
    duy_max: float = field(default = 1.5)
    ''' max steering angle rate in radians per second '''
    duy_min: float = field(default =-1.5)
    ''' min steering angle rate in radians per second '''

    ud_max: float = field(default = 0.5)
    ''' max rider tilt accel in m/s^2 '''
    ud_min: float = field(default =-0.5)
    ''' min rider tilt accel in m/s^2 '''
    dud_max: float = field(default = 1.5)
    ''' max rider tilt jerk in m/s^3 '''
    dud_min: float = field(default =-1.5)
    ''' min rider tilt jerk in m/s^3 '''

    b1: float = field(default = 0.)
    ''' linear drag term, force located at COM when d = 0 '''
    c1: float = field(default = 0.)
    ''' quadratic drag term, force located at COM when d = 0 '''
    c3: float = field(default = 0.)
    ''' quadratic down force term, force located at COM when d = 0 '''

    P_max: float = field(default = 5e4)
    ''' max power output of the rear wheel '''

@dataclass
class MotorcycleModelVars(DAEDynamicsModelVars):
    '''
    variables for motorcycle model
    forces are internally normalized by m*g
    '''
    uFf: ca.SX = field(default = None)
    uFr: ca.SX = field(default = None)
    uy: ca.SX = field(default = None)
    ud_ddot: ca.SX = field(default = None)

    s: ca.SX = field(default = None)
    s_dot: ca.SX = field(default = None)
    y: ca.SX = field(default = None)
    y_dot: ca.SX = field(default = None)
    n: ca.SX = field(default = None)
    n_dot: ca.SX = field(default = None)
    Rb: ca.SX = field(default = None)

    ths: ca.SX = field(default = None)
    ths_dot: ca.SX = field(default = None)

    d: ca.SX = field(default = None)
    d_dot: ca.SX = field(default = None)
    c: ca.SX = field(default = None)
    c_dot: ca.SX = field(default = None)
    c_ddot: ca.SX = field(default = None)

    front_tire_N_normalized: ca.SX = field(default = None)
    front_tire_N: ca.SX = field(default = None)
    front_tire_c: ca.SX = field(default = None)
    front_tire_y: ca.SX = field(default = None)
    rear_tire_N_normalized: ca.SX = field(default = None)
    rear_tire_N: ca.SX = field(default = None)
    rear_tire_c: ca.SX = field(default = None)
    rear_tire_y: ca.SX = field(default = None)

    v1: ca.SX = field(default = None)
    v2: ca.SX = field(default = None)
    v3: ca.SX = field(default = None)

    w1: ca.SX = field(default = None)
    w2: ca.SX = field(default = None)
    w3: ca.SX = field(default = None)

    v1_dot: ca.SX = field(default = None)
    v2_dot: ca.SX = field(default = None)
    v3_dot: ca.SX = field(default = None)

    w1_dot: ca.SX = field(default = None)
    w2_dot: ca.SX = field(default = None)
    w3_dot: ca.SX = field(default = None)

    F: ca.SX = field(default = None)
    K: ca.SX = field(default = None)
    Fg: ca.SX = field(default = None)
    Fd: ca.SX = field(default = None)
    Kd: ca.SX = field(default = None)
    Ft: ca.SX = field(default = None)
    Kt: ca.SX = field(default = None)
    F_required: ca.SX = field(default = None)
    K_required: ca.SX = field(default = None)

    R_c: ca.SX = field(default = None)
    eb: ca.SX = field(default = None)
    em: ca.SX = field(default = None)
    r_com: ca.SX = field(default = None)
    wm: ca.SX = field(default = None)
    eb_dot: ca.SX = field(default = None)

    frame_instances: ca.SX = field(default = None)
    com_frame: ca.SX = field(default = None)


class MotorcycleDynamics(DAEDynamicsModel):
    '''
    dynamics of the motorcycle
    only the rear wheel is powered but both wheels may brake independently
    '''
    config: MotorcycleConfig
    model_vars: MotorcycleModelVars
    tire_model: MotorcycleTireModel
    front_tire: MotorcycleTire
    rear_tire: MotorcycleTire
    tires: List[MotorcycleTire]

    f_F: ca.Function
    f_K: ca.Function

    f_ths : ca.Function
    f_d : ca.Function
    f_c : ca.Function
    f_c_dot : ca.Function
    f_d_dot : ca.Function
    frame_instances: ca.Function

    # functions to unpack tire state
    f_tf: ca.Function
    f_tr: ca.Function

    # rendering functions
    f_fork_instances: ca.Function
    f_N_instances: ca.Function
    f_Ft_instances: ca.Function
    f_com_instances: ca.Function

    f_Rb: ca.Function

    def __init__(self, config: MotorcycleConfig, surf: BaseSurface):
        self.tire_model = MotorcycleTireModel(config.tire_config)
        super().__init__(config, surf)

    def _create_model(self):
        self._clear_model_vars()
        self._add_input_vars()
        self._add_pose_vars()
        self._add_vel_vars()
        if self.model_vars.wb is None or self. model_vars.s_dot is None:
            raise RuntimeError('Velocity setup not completed, likely a developer issue')
        self._add_forces()
        self._state_derivative()
        self._calc_outputs()

    def _clear_model_vars(self):
        self.model_vars = MotorcycleModelVars()

    def _add_input_vars(self):
        self.model_vars.uFf = ca.SX.sym('uFf')
        self.model_vars.uFr = ca.SX.sym('uFr')
        self.model_vars.uy = ca.SX.sym('uy')
        self.model_vars.ud_ddot = ca.SX.sym('ud_ddot')
        self.model_vars.u = ca.vertcat(
            self.model_vars.uFf,
            self.model_vars.uFr,
            self.model_vars.uy,
            self.model_vars.ud_ddot)

    def _add_pose_vars(self):
        self.model_vars.s = self.surf.sym_rep.s
        self.model_vars.y = self.surf.sym_rep.y
        self.model_vars.n = self.config.tire_rad
        self.model_vars.ths = self.surf.sym_rep.ths
        self.model_vars.Rb = self.surf.sym_rep.R_ths

        self.model_vars.p = ca.vertcat(
            self.model_vars.s,
            self.model_vars.y,
            self.model_vars.n
        )

        c = ca.SX.sym('c')
        d = ca.SX.sym('d')
        self.model_vars.c = c
        self.model_vars.d = d
        R_c = ca.horzcat(
            ca.vertcat(1, 0, 0),
            ca.vertcat(0,ca.cos(c), -ca.sin(c)),
            ca.vertcat(0, ca.sin(c), ca.cos(c))
        )
        self.model_vars.R_c = R_c
        self.model_vars.R = self.model_vars.Rb @ R_c

        eb = ca.SX.sym('eb',3)
        em = R_c.T @ eb
        r_com = em[1] * d + em[2] * (self.config.h - self.config.tire_rad)
        self.model_vars.eb = eb
        self.model_vars.em = em
        self.model_vars.r_com = r_com

        head = self.config.head_angle
        R_head = ca.horzcat(
            ca.vertcat(ca.cos(head), 0, ca.sin(head)),
            ca.vertcat(0,1,0),
            ca.vertcat(-ca.sin(head), 0, ca.cos(head))
        )

        uy = self.model_vars.uy
        R_u = ca.horzcat(
            ca.vertcat(ca.cos(uy), ca.sin(uy), 0),
            ca.vertcat(-ca.sin(uy),ca.cos(uy),0),
            ca.vertcat(0,0,1)
        )

        R_front = R_head @ R_u
        R_front_from_b = R_c @ R_front
        x_front = ca.vertcat(self.config.lf, 0, 0) + \
            self.config.rake * R_front[:,0]

        self.model_vars.rear_tire_c = c
        self.model_vars.front_tire_c = -ca.arcsin(R_front_from_b[2,1])
        self.model_vars.rear_tire_y = 0
        self.model_vars.front_tire_y = ca.arctan2(-R_front_from_b[0,1],  R_front_from_b[1,1])

        # set up symbolics for visualizing frames
        front_frame = ca.SX_eye(4)
        front_frame[:3,:3] = R_head @ R_u
        front_frame[:3,-1] = x_front
        rear_frame = ca.SX_eye(4)
        rear_frame[:3,-1] = ca.vertcat(-self.config.lr, 0, 0)
        cent_frame = ca.SX_eye(4)
        cent_frame[:3,-1] = ca.vertcat(0, 0, self.config.h - self.config.tire_rad)

        self.model_vars.frame_instances = ca.vertcat(front_frame,rear_frame,cent_frame)
        self.model_vars.com_frame = ca.SX_eye(4)
        self.model_vars.com_frame[:3,-1] = ca.vertcat(0, d, self.config.h - self.config.tire_rad)

    def _add_vel_vars(self):
        v1 = ca.SX.sym('v1')
        v2 = ca.SX.sym('v2')
        v3 = 0
        w3 = ca.SX.sym('w3')
        self.model_vars.v1 = v1
        self.model_vars.v2 = v2
        self.model_vars.w3 = w3

        if self.config.build_planar_model:
            s_dot, y_dot, ths_dot, w1, w2 = self.surf.pose_eqns_2D_planar(
                vb1  = self.model_vars.v1,
                vb2  = self.model_vars.v2,
                wb3  = self.model_vars.w3,
                n   = self.model_vars.n)
        else:
            s_dot, y_dot, ths_dot, w1, w2 = self.surf.pose_eqns_2D(
                vb1  = self.model_vars.v1,
                vb2  = self.model_vars.v2,
                wb3  = self.model_vars.w3,
                n   = self.model_vars.n)

        self.model_vars.s_dot = s_dot
        self.model_vars.y_dot = y_dot
        self.model_vars.n_dot = 0
        self.model_vars.ths_dot = ths_dot
        self.model_vars.w1 = w1
        self.model_vars.w2 = w2
        self.model_vars.v3 = v3
        self.model_vars.v3_dot = 0

        self.model_vars.vb = ca.vertcat(v1, v2, v3)
        self.model_vars.wb = ca.vertcat(w1, w2, w3)

        self.model_vars.c_dot = ca.SX.sym('c_dot')
        self.model_vars.d_dot = ca.SX.sym('d_dot')

        self.model_vars.eb_dot = -hat(self.model_vars.wb) @ self.model_vars.eb
        self.model_vars.wm = self.model_vars.R_c.T @ self.model_vars.wb \
            + ca.vertcat(self.model_vars.c_dot, 0, 0)

    def _add_forces(self):
        self._grav_forces()
        self._drag_forces()
        self._tire_forces()
        self.model_vars.F = self.model_vars.Fg + self.model_vars.Fd + self.model_vars.Ft
        self.model_vars.K = self.model_vars.Kd + self.model_vars.Kt

    def _grav_forces(self):
        if self.surf.config.flat or self.config.build_planar_model:
            Fg1 = 0
            Fg2 = 0
            Fg3 = -self.config.m * self.config.g
        else:
            R = self.model_vars.Rb
            e1 = R[:,0]
            e2 = R[:,1]
            e3 = R[:,2]
            Fg1 = -self.config.m * self.config.g * e1[2]
            Fg2 = -self.config.m * self.config.g * e2[2]
            Fg3 = -self.config.m * self.config.g * e3[2]

        self.model_vars.Fg = ca.vertcat(Fg1, Fg2, Fg3)

    def _drag_forces(self):
        v1 = self.model_vars.vb[0]

        F_drag = -self.config.c1 * v1 * ca_abs(v1) \
            - self.config.b1 * v1
        F_down = -self.config.c3 * v1 * v1
        self.model_vars.Fd = self.model_vars.R_c @ ca.vertcat(F_drag, 0, F_down)
        self.model_vars.Kd = ca.vertcat(0, 0, 0)

    def _tire_forces(self):
        vb = self.model_vars.vb
        wb = self.model_vars.wb

        self.model_vars.front_tire_N_normalized = ca.SX.sym('Nf')
        self.model_vars.rear_tire_N_normalized = ca.SX.sym('Nr')

        self.model_vars.front_tire_N = self.model_vars.front_tire_N_normalized \
            * self.config.m * self.config.g
        self.model_vars.rear_tire_N = self.model_vars.rear_tire_N_normalized \
            * self.config.m * self.config.g

        x_b_com = ca.jacobian(self.model_vars.r_com, self.model_vars.eb).T
        x_front_b = ca.vertcat(self.config.lf, 0, -self.config.tire_rad)
        x_rear_b = ca.vertcat(-self.config.lr, 0, -self.config.tire_rad)

        self.front_tire = self.tire_model.apply_model(
            x = x_front_b,
            x_com = x_front_b + x_b_com,
            vb = vb,
            wb = wb,
            Fz = self.model_vars.front_tire_N,
            Fx = self.model_vars.uFf * self.config.m * self.config.g,
            y = self.model_vars.front_tire_y,
            c = self.model_vars.front_tire_c,
            rear = False
        )
        self.rear_tire = self.tire_model.apply_model(
            x = x_rear_b,
            x_com = x_rear_b + x_b_com,
            vb = vb,
            wb = wb,
            Fz = self.model_vars.rear_tire_N,
            Fx = self.model_vars.uFr * self.config.m * self.config.g,
            y = self.model_vars.rear_tire_y,
            c = self.model_vars.rear_tire_c,
            rear = True
        )

        self.model_vars.Ft = self.front_tire.Fb + self.rear_tire.Fb
        self.model_vars.Kt = self.front_tire.Kb + self.rear_tire.Kb

    def _state_derivative(self):
        ''' body dynamics '''
        m = self.config.m

        d = self.model_vars.d
        d_dot = self.model_vars.d_dot
        d_ddot = self.model_vars.ud_ddot

        c = self.model_vars.c
        c_dot = self.model_vars.c_dot
        c_ddot = ca.SX.sym('c_ddot')

        vb = self.model_vars.vb
        v1_dot = ca.SX.sym('v1_dot')
        v2_dot = ca.SX.sym('v2_dot')
        vb_dot = ca.vertcat(v1_dot, v2_dot, 0)

        wb = self.model_vars.wb
        one = self.surf.sym_rep.one
        two = self.surf.sym_rep.two
        n = self.model_vars.n
        J = self.surf.sym_rep.J
        J_inv = self.surf.sym_rep.J_inv
        dw1w2 = J_inv @ two @ ca.inv(one - n * two) @ J @ ca.vertcat(v1_dot,v2_dot)
        w1_dot =  dw1w2[1]
        w2_dot = -dw1w2[0]
        w3_dot = ca.SX.sym('w3_dot')
        wb_dot = ca.vertcat(w1_dot, w2_dot, w3_dot)

        self.model_vars.d_dot = d_dot
        self.model_vars.c_dot = c_dot
        self.model_vars.c_ddot = c_ddot
        self.model_vars.v1_dot = v1_dot
        self.model_vars.v2_dot = v2_dot
        self.model_vars.w1_dot = w1_dot
        self.model_vars.w2_dot = w2_dot
        self.model_vars.w3_dot = w3_dot


        z = ca.vertcat(
            self.model_vars.s,
            self.model_vars.y,
            self.model_vars.ths,
            self.model_vars.v1,
            self.model_vars.v2,
            self.model_vars.w3,
            self.model_vars.c,
            self.model_vars.c_dot,
            self.model_vars.d,
            self.model_vars.d_dot,
        )

        a = ca.vertcat(
            self.model_vars.c_ddot,
            self.model_vars.v1_dot,
            self.model_vars.v2_dot,
            self.model_vars.w3_dot,
            self.model_vars.front_tire_N_normalized,
            self.model_vars.rear_tire_N_normalized
        )

        z_dot = ca.vertcat(
            self.model_vars.s_dot,
            self.model_vars.y_dot,
            self.model_vars.ths_dot,
            self.model_vars.v1_dot,
            self.model_vars.v2_dot,
            self.model_vars.w3_dot,
            self.model_vars.c_dot,
            self.model_vars.c_ddot,
            self.model_vars.d_dot,
            self.model_vars.ud_ddot,
        )

        self.model_vars.z = z
        self.model_vars.z_dot = z_dot
        self.model_vars.a = a

        # compute dynamics in DAE form

        # force
        eb = self.model_vars.eb
        em = self.model_vars.em
        wm = self.model_vars.wm
        r_com = self.model_vars.r_com
        eb_dot = self.model_vars.eb_dot

        v_com = vb.T @ eb + \
            ca.jacobian(r_com, eb) @ eb_dot + \
            ca.jacobian(r_com, d) @ d_dot + \
            ca.jacobian(r_com, c) @ c_dot

        v_dot = ca.jacobian(v_com, vb[:2]) @ vb_dot[:2] + \
            ca.jacobian(v_com, wb[2]) @ wb_dot[2] + \
            ca.jacobian(v_com, eb) @ eb_dot + \
            ca.jacobian(v_com, d) @ d_dot + \
            ca.jacobian(v_com, d_dot) @ d_ddot + \
            ca.jacobian(v_com, c) @ c_dot + \
            ca.jacobian(v_com, c_dot) @ c_ddot
        p_dot = m * v_dot

        self.model_vars.F_required = ca.jacobian(p_dot, eb).T

        # moment
        l_rear_wheel = em[1] * self.tire_model.config.Iyy * self.rear_tire.w
        l_front_wheel = em[1] * self.tire_model.config.Iyy * self.front_tire.w
        l = em.T @ ca.diag((self.config.I1, self.config.I2, self.config.I3)) @ wm + \
            l_rear_wheel + \
            l_front_wheel
        l_dot = ca.jacobian(l, vb[:2]) @ vb_dot[:2] + \
            ca.jacobian(l, wb[2]) @ wb_dot[2] + \
            ca.jacobian(l, eb) @ eb_dot + \
            ca.jacobian(l, d) @ d_dot + \
            ca.jacobian(l, d_dot) @ d_ddot + \
            ca.jacobian(l, c) @ c_dot + \
            ca.jacobian(l, c_dot) @ c_ddot

        self.model_vars.K_required = ca.jacobian(l_dot, eb).T

        self.model_vars.h = ca.vertcat(
            self.model_vars.F - self.model_vars.F_required,
            self.model_vars.K - self.model_vars.K_required,
        )

        # body frame acceleration
        a1 = v1_dot - self.model_vars.w3*self.model_vars.v2 - self.model_vars.Fg[0] / self.config.m
        a2 = v2_dot + self.model_vars.w3*self.model_vars.v1 - self.model_vars.Fg[1] / self.config.m
        a3 = self.model_vars.v2*self.model_vars.w1 - self.model_vars.v1*self.model_vars.w2 \
            - self.model_vars.Fg[2] / self.config.m

        self.model_vars.v1_dot = v1_dot
        self.model_vars.v2_dot = v2_dot
        self.model_vars.w3_dot = w3_dot
        self.model_vars.ab = ca.vertcat(a1, a2, a3)

    def _calc_outputs(self):
        self.model_vars.g = ca.vertcat(
            self.rear_tire.P / self.config.P_max,
            self.model_vars.uFf - self.model_vars.front_tire_N_normalized,
            self.model_vars.uFf + self.model_vars.front_tire_N_normalized,
            self.model_vars.uFr - self.model_vars.rear_tire_N_normalized,
            self.model_vars.uFr + self.model_vars.rear_tire_N_normalized,
        )
        self.model_vars.ubg = [
            1.0,
            0.0,
            np.inf,
            0.0,
            np.inf
        ]
        self.model_vars.lbg = [
            -np.inf,
            -np.inf,
            0.0,
            -np.inf,
            0.0
        ]

    def _setup_helper_functions(self):
        super()._setup_helper_functions()
        self.f_ths = ca.Function('ths',[self.model_vars.z],[self.model_vars.ths])
        self.f_d = ca.Function('d',[self.model_vars.z],[self.model_vars.d])
        self.f_c = ca.Function('c',[self.model_vars.z],[self.model_vars.c])
        self.f_c_dot = ca.Function('c_dot',[self.model_vars.z],[self.model_vars.c_dot])
        self.f_d_dot = ca.Function('d_dot',[self.model_vars.z],[self.model_vars.d_dot])
        self.f_Rb = self.surf.fill_in_param_terms([self.model_vars.Rb], [self.model_vars.z], )

        self.f_F = self.surf.fill_in_param_terms(
            [self.model_vars.F],
            self.model_vars.get_all_indep_vars(),
        )
        self.f_K = self.surf.fill_in_param_terms(
            [self.model_vars.K],
            self.model_vars.get_all_indep_vars(),
        )

        self.f_tf = self.surf.fill_in_param_terms(
            [self.front_tire.get_tire_state_vec()],
            self.model_vars.get_all_indep_vars(),
        )
        self.f_tr = self.surf.fill_in_param_terms(
            [self.rear_tire.get_tire_state_vec()],
            self.model_vars.get_all_indep_vars()
        )

    def get_empty_state(self):
        state = BaseMotorcycleState()
        state.tf.N = self.config.lr / (self.config.lr + self.config.lr) \
            * self.config.m * self.config.g
        state.tr.N = self.config.lf / (self.config.lr + self.config.lr) \
            * self.config.m * self.config.g
        return state

    def state2u(self, state: BaseMotorcycleState):
        u = state.u.to_vec().tolist()
        u[0] /= self.config.m * self.config.g
        u[1] /= self.config.m * self.config.g
        return u

    def state2z(self, state: BaseMotorcycleState):
        return [
            state.p.s,
            state.p.y,
            state.ths,
            state.vb.v1,
            state.vb.v2,
            state.wb.w3,
            state.c,
            state.c_dot,
            state.d,
            state.d_dot
        ]

    def state2a(self, state: BaseMotorcycleState):
        return [
            state.c_ddot,
            state.v1_dot,
            state.v2_dot,
            state.w3_dot,
            state.tf.N / self.config.m / self.config.g,
            state.tr.N / self.config.m / self.config.g
        ]

    def u2state(self, state: BaseMotorcycleState, u: np.ndarray):
        state.u.from_vec([
            u[0] * self.config.m * self.config.g,
            u[1] * self.config.m * self.config.g,
            *u[2:],
            ])

    def zua2state(self, state: BaseMotorcycleState, z, u, a):
        super().zua2state(state, z, u, a)
        state.ths = float(self.f_ths(z))
        state.d = float(self.f_d(z))
        state.c = float(self.f_c(z))
        state.c_dot = float(self.f_c_dot(z))
        state.d_dot = float(self.f_d_dot(z))

        state.c_ddot = float(a[0])
        state.v1_dot = float(a[1])
        state.v2_dot = float(a[2])
        state.w3_dot = float(a[3])

        state.F.from_vec(self.f_F(z,u,a))
        state.K.from_vec(self.f_K(z,u,a))

        state.tf.from_vec(self.f_tf(z,u,a))
        state.tr.from_vec(self.f_tr(z,u,a))

    def uu(self):
        return [
            0,
            1,
            self.config.uy_max,
            self.config.ud_max,
        ]

    def ul(self):
        return [
            -1,
            -1,
            self.config.uy_min,
            self.config.ud_min,
        ]

    def duu(self):
        return [
            2,
            2,
            self.config.duy_max,
            self.config.dud_max,
        ]

    def dul(self):
        return [
            -2,
            -2,
            self.config.duy_min,
            self.config.dud_min,
        ]

    def zu(self, s: float = 0.):
        return [
            self.surf.s_max(),
            self.surf.y_max(s),
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            self.config.c_max,
            np.inf,
            self.config.d_max,
            np.inf,
        ]

    def zl(self, s: float = 0.):
        return [
            self.surf.s_min(),
            self.surf.y_min(s),
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            self.config.c_min,
            -np.inf,
            self.config.d_min,
            -np.inf,
        ]

    def au(self):
        return [np.inf] * 6

    def al(self):
        return [-np.inf] * 4 + [1e-2, 1e-2]

    def get_color(self):
        return [.5, 1, 0, 1]

    def get_label(self):
        if self.config.build_planar_model:
            return 'Planar Motorcycle'
        return 'Motorcycle'

    def generate_visual_assets(self, ubo: UBOObject) -> Dict[str, VertexObject]:
        ''' visualize the motorcycle '''
        #TODO - dynamic sizing
        if self.visual_assets is not None:
            return self.visual_assets

        body, fork = load_split_motorcycle(ubo, color = self.get_color())
        R_fork = ca.DM_eye(4)
        R_fork = rotate(R_fork, -180/np.pi*self.config.head_angle, 0, 1, 0)
        R_fork = rotate(R_fork, 180/np.pi*self.model_vars.uy,      0, 0, 1)
        R_fork = rotate(R_fork, 180/np.pi*self.config.head_angle,  0, 1, 0)
        d = self.config.lf + 0.25
        R_fork = translate_rel(R_fork, -d, 0, 0)
        R_fork = translate(R_fork, d, 0, 0)
        self.f_fork_instances = self.surf.fill_in_param_terms(
            [R_fork],
            self.model_vars.get_all_indep_vars()
        )

        V, I = join_vis((
            get_unit_arrow(d=1, color = [0, 1, 1, 1]),
            get_unit_arrow(d=2, color = [0, 1, 0, 1]),
            get_unit_arrow(d=3, color = [0, 0, 1, 1])
        ))
        frames = InstancedVertexObject(ubo, V, I)

        V, I = get_unit_arrow(d=3, color = [1, 0, 0, 1])
        normal_forces = InstancedVertexObject(ubo, V, I)
        normal_force_instances = ca.vertcat(
            self.front_tire.normal_force_instance_matrix,
            self.rear_tire.normal_force_instance_matrix
        )
        self.f_N_instances = self.surf.fill_in_param_terms(
            normal_force_instances,
            self.model_vars.get_all_indep_vars()
        )

        # get tire force object and function to compute instance transforms
        tire_forces = InstancedVertexObject(ubo, V, I)
        tire_foce_instances = ca.vertcat(
            self.front_tire.tire_force_instance_matrix,
            self.rear_tire.tire_force_instance_matrix
        )
        self.f_Ft_instances = self.surf.fill_in_param_terms(
            tire_foce_instances,
            self.model_vars.get_all_indep_vars()
        )

        V, I = get_sphere(r = 0.2, color = [0,0,0,1])
        com_instances = InstancedVertexObject(ubo, V, I)
        self.f_com_instances = self.surf.fill_in_param_terms(
            [self.model_vars.com_frame],
            self.model_vars.get_all_indep_vars()
        )

        self.visual_assets = {
            'Motorcycle Body': body,
            'Motorcycle Fork': fork,
            'Frames': frames,
            'Normal Forces': normal_forces,
            'Tire Forces': tire_forces,
            'COM': com_instances
        }
        self.frame_instances = self.surf.fill_in_param_terms(
            self.model_vars.frame_instances,
            self.model_vars.get_all_indep_vars()
        )
        return self.visual_assets

    def update_visual_assets(self, state: BaseMotorcycleState, dt: float = None):
        self.visual_assets['Motorcycle Body'].update_pose(state.x, state.q)
        self.visual_assets['Motorcycle Fork'].update_pose(state.x, state.q)
        self.visual_assets['Frames'].update_pose(state.x, state.q)
        self.visual_assets['COM'].update_pose(state.x, state.q)
        Rb = np.eye(4, dtype = np.float32)
        Rb[:3,:3] = self.f_Rb(self.state2z(state))
        Rb[:3,-1] = state.x.to_vec()
        self.visual_assets['Normal Forces'].update_pose(mat = Rb)
        self.visual_assets['Tire Forces'].update_pose(mat = Rb)
        z,u,a = self.state2zua(state)
        self.visual_assets['Frames'].apply_instancing(
            self.frame_instances(z,u,a).reshape((-1,4,4)).astype(np.float32)
        )
        self.visual_assets['Normal Forces'].apply_instancing(
            np.array(self.f_N_instances(z,u,a)).reshape((-1,4,4)).astype(np.float32)
        )
        self.visual_assets['Tire Forces'].apply_instancing(
            np.array(self.f_Ft_instances(z,u,a)).reshape((-1,4,4)).astype(np.float32)
        )
        self.visual_assets['COM'].apply_instancing(
            np.array(self.f_com_instances(z,u,a)).reshape((-1,4,4)).astype(np.float32)
        )
        self.visual_assets['Motorcycle Fork'].apply_instancing(
            np.array(self.f_fork_instances(z,u,a)).reshape((-1,4,4)).astype(np.float32)
        )

    def get_instanced_visual_asset(self, ubo: UBOObject):
        #TODO - dynamic sizing
        return load_motorcycle(ubo, color = self.get_color(), instanced = True)
