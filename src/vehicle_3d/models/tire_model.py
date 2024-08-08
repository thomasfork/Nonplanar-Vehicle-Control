'''
simplified magic formula tire model
no camber angle or turn slip, meant for four-wheeled vehicles
'''
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import numpy as np
import casadi as ca

from vehicle_3d.pytypes import PythonMsg, VectorizablePythonMsg
from vehicle_3d.utils.ca_utils import hat, ca_pos_abs, ca_smooth_sign
from vehicle_3d.visualization.ca_glm import rotate, translate
from vehicle_3d.visualization.utils import get_thrust_tf_ca

class TireInputChoices(Enum):
    ''' different input choices to a wheel '''
    FREE_SPIN = -1
    LONGITUDINAL_FORCE = 0
    LONGITUDINAL_SLIP  = 1
    WHEEL_TORQUE = 2
    WHEEL_BRAKE_TORQUE = 3


class TireFlags(Enum):
    ''' flags for different wheel / tire options '''
    NO_FORCES = 0
    LINEAR_SLIP_MODEL = 1

@dataclass
class TireConfig(PythonMsg):
    '''
    configuration class for a Pacjeka tire

    when evaluated in a linear manner:
    Fx = Bx * s
    Fy = By * a
    assuming all other coefficients ~ 1
    so that Bx, By are the cornering stiffness
    '''
    m: float = field(default = 8)
    I: float = field(default = 1)
    mu: float = field(default = 1.0)

    Bx: float = field(default = 16)
    Cx: float = field(default = 1.58)
    Ex: float = field(default = 0.1)
    By: float = field(default = 13)
    Cy: float = field(default = 1.45)
    Ey: float = field(default = -0.8)

    Exa: float = field(default = -0.5)
    Cxa: float = field(default = 1)
    rBx1: float = field(default = 13)
    rBx2: float = field(default = 9.7)

    Eys: float = field(default = 0.3)
    Cys: float = field(default = 1)
    rBy1: float = field(default = 10.62)
    rBy2: float = field(default = 7.82)

    r:  float = field(default =    0.3)     # tire geometric radius to ground

    # slip input and input rate limits (for slip input models)
    s_max: float = field(default = 1)
    s_min: float = field(default =-1)
    ds_max: float = field(default = 2)
    ds_min: float = field(default =-2)

    def longitudinal_stiffness(self):
        ''' longitudinal stiffness with mu*N term omitted '''
        return self.Cx * self.Bx

    def lateral_stiffness(self):
        ''' latearl stiffness with mu*N term omitted '''
        return self.Cy * self.By

    def combined_slip_forces(self, s: ca.SX, a: ca.SX, N: ca.SX, mu: ca.SX, linear: bool = False):
        '''
        computes combined slip lateral/longitudinal tire forces using
            s: slip ratio
            a: slip angle
            N: normal force
        '''
        if linear:
            Fx = mu * N * self.Cx * self.Bx * s
            Fy = mu * N * self.Cy * self.By * a
            return Fx, Fy

        Fx0 = mu * N * ca.sin(self.Cx * ca.arctan((1 - self.Ex) * self.Bx * s +
                                                    self.Ex * ca.arctan(self.Bx * s)))
        Fy0 = mu * N * ca.sin(self.Cy * ca.arctan((1 - self.Ey) * self.By * a +
                                                    self.Ey * ca.arctan(self.By * a)))

        Bxa = self.rBx1 * ca.cos(ca.arctan(self.rBx2 * s))
        Gxa = ca.cos(self.Cxa * ca.arctan(Bxa * a - self.Exa * (Bxa * a - ca.arctan(Bxa * a))))

        Bys = self.rBy1 * ca.cos(ca.arctan(self.rBy2 * a))
        Gys = ca.cos(self.Cys * ca.arctan(Bys * s - self.Eys * (Bys * s - ca.arctan(Bys * s))))

        Fx  = Fx0 * Gxa
        Fy  = Fy0 * Gys
        return Fx,Fy

    def Fx_input_forces(self, Fx: ca.SX, a: ca.SX, N: ca.SX, mu: ca.SX, linear: bool = False):
        '''
        computes combined slip lateral/longitudinal tire forces using
            Fx: given longitudinal force (must be less than mu*N in magnitude)
            a: slip angle
            N: normal force
        '''
        if linear:
            Fy = mu * N * self.Cy * self.By * a
            return Fx, Fy

        Fy = ca.sqrt(mu**2*N**2 - Fx**2) * ca.sin(self.Cy * ca.arctan((1 - self.Ey) * self.By * a +
                                                    self.Ey * ca.arctan(self.By * a)))

        return Fx,Fy

@dataclass
class TireState(VectorizablePythonMsg):
    ''' state of a single tire on the car '''
    N: float = field(default = 0.)
    ''' tire normal force'''
    T: float = field(default = 0.)
    ''' wheel torque '''
    Fx: float = field(default = 0.)
    ''' longitudinal tire force '''
    y: float = field(default = 0.)
    ''' steering angle: yaw angle relative to road normal '''
    s: float = field(default = 0.)
    ''' slip ratio '''
    a: float = field(default = 0.)
    ''' slip angle'''
    c: float = field(default = 0.)
    ''' camber angle '''
    w: float = field(default = 0.)
    ''' tire spin angular velocity '''
    th: float = field(default = 0.)
    ''' tire spin rotational angle'''

    def to_vec(self):
        return np.array([self.N, self.T, self.Fx, self.y,
            self.s, self.a, self.c, self.w])

    def from_vec(self, vec):
        self.N, self.T, self.Fx, self.y, \
            self.s, self.a, self.c, self.w = vec

@dataclass
class Tire(PythonMsg):
    '''
    class to represent a tire that has been created
    stores net force and moment on the body frame as well as
    matrices for instanced rendering of tire forces
    '''
    N: ca.SX = field(default = None)
    T: ca.SX = field(default = None)
    Fx: ca.SX = field(default = None)
    y: ca.SX = field(default = None)
    s: ca.SX = field(default = None)
    a: ca.SX = field(default = None)
    c: ca.SX = field(default = None)
    w: ca.SX = field(default = None)
    th: ca.SX = field(default = None)
    w_dot: ca.SX = field(default = None)
    Kt: ca.SX = field(default = None)
    Fb: ca.SX = field(default = None)
    Kb: ca.SX = field(default = None)
    normal_force_instance_matrix: ca.SX = field(default = None)
    tire_force_instance_matrix: ca.SX = field(default = None)

    def __post_init__(self):
        for _, item in vars(self).items():
            assert item is not None

    def get_tire_state_vec(self):
        ''' return tire state vectors (vectorized TireState)'''
        return ca.vertcat(
            self.N,
            self.T,
            self.Fx,
            self.y,
            self.s,
            self.a,
            self.c,
            self.w
        )

def get_tire(config: TireConfig,
        x: ca.SX,
        vb: ca.SX,
        wb: ca.SX,
        mu: ca.SX,
        N: ca.SX,
        y: ca.SX,
        input_mode: TireInputChoices,
        flags: List[TireFlags] = None,
        visual_force_norm: float = 2000
        )\
    -> Tire:
    ''' get a tire model as a Tire structure '''
    if flags is None:
        flags = []

    M = rotate(ca.DM_eye(4), y*180 / ca.pi, 0, 0, 1)
    M = translate(M, x[0], x[1], x[2])
    # tire frame velocity at tire contact patch
    vt = M[:3,:3].T @ (vb + hat(wb) @ x)

    # slip angle
    alpha = -ca.arctan(vt[1] / ca_pos_abs(vt[0]))

    # slip ratio
    if input_mode == TireInputChoices.FREE_SPIN:
        sigma = 0
        w = vt[0] / config.r
        th = 0
        T = 0
    elif input_mode == TireInputChoices.LONGITUDINAL_SLIP:
        sigma = ca.SX.sym('sigma')
        w = vt[0] / config.r
        th = 0
        T = 0
    elif input_mode == TireInputChoices.LONGITUDINAL_FORCE:
        Fx = ca.SX.sym('Fx')
        sigma = 0
        w = vt[0] / config.r
        th = 0
        T = 0
    elif input_mode == TireInputChoices.WHEEL_TORQUE:
        T = ca.SX.sym('T')
        w = ca.SX.sym('w')
        th = ca.SX.sym('th')
        sigma = - (vt[0] - w * config.r) / ca_pos_abs(vt[0])
    elif input_mode == TireInputChoices.WHEEL_BRAKE_TORQUE:
        T = ca.SX.sym('T')
        w = ca.SX.sym('w')
        th = ca.SX.sym('th')
        sigma = - (vt[0] - w * config.r) / ca_pos_abs(vt[0])
    else:
        raise NotImplementedError()

    # the in-plane tire forces
    if input_mode == TireInputChoices.LONGITUDINAL_FORCE:
        Fx,Fy = config.Fx_input_forces(Fx, alpha, N, mu,
            linear = TireFlags.LINEAR_SLIP_MODEL in flags)
    else:
        Fx,Fy = config.combined_slip_forces(sigma, alpha, N, mu,
            linear = TireFlags.LINEAR_SLIP_MODEL in flags)

    if input_mode == TireInputChoices.WHEEL_TORQUE:
        w_dot = (-Fx * config.r + T) / config.I
    elif input_mode == TireInputChoices.WHEEL_BRAKE_TORQUE:
        w_dot = (-Fx * config.r - T * ca_smooth_sign(w)) / config.I
    else:
        w_dot = 0

    Ft = ca.vertcat(Fx, Fy, N)
    Kt = Fx / config.r
    Fb = M[:3,:3] @ Ft
    Kb = hat(x) @ Fb

    # tire forces as global frame vectors
    Fx_b = M[:3,0] * Fx
    Fy_b = M[:3,1] * Fy
    Fz_b = M[:3,2] * N

    normal_force_instance_matrix = translate(ca.SX_eye(4), x[0], x[1], x[2])
    normal_force_instance_matrix[:3,:3] = get_thrust_tf_ca(Fz_b, norm=visual_force_norm)
    Fx_instance_matrix = translate(ca.SX_eye(4), x[0], x[1], x[2])
    Fx_instance_matrix[:3,:3] = get_thrust_tf_ca(Fx_b, norm=visual_force_norm)
    Fy_instance_matrix = translate(ca.SX_eye(4), x[0], x[1], x[2])
    Fy_instance_matrix[:3,:3] = get_thrust_tf_ca(Fy_b, norm=visual_force_norm)

    if TireFlags.NO_FORCES in flags:
        tire_force_instance_matrix = ca.vertcat([])
    elif input_mode == TireInputChoices.FREE_SPIN:
        tire_force_instance_matrix = ca.vertcat(
            Fy_instance_matrix,
        )
    else:
        tire_force_instance_matrix = ca.vertcat(
            Fx_instance_matrix,
            Fy_instance_matrix,
        )

    return Tire(
        N = N,
        T = T,
        Fx = Fx,
        y = y,
        s = sigma,
        a = alpha,
        c = 0., #TODO
        w = w,
        th = th,
        w_dot = w_dot,
        Kt = Kt,
        Fb = Fb,
        Kb = Kb,
        normal_force_instance_matrix = normal_force_instance_matrix,
        tire_force_instance_matrix = tire_force_instance_matrix,
    )
