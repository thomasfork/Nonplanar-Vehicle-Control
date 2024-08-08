'''
motorcycle tire model, equations and coefficients adapted from
Chapter 11 of "Tire and Vehicle Dyanmcis" 3rd Edition by Hans Pacejcka

longitudinal tire force is treated as an input
'''
from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from vehicle_3d.pytypes import PythonMsg, VectorizablePythonMsg
from vehicle_3d.utils.ca_utils import hat, ca_pos_abs
from vehicle_3d.visualization.ca_glm import rotate, translate
from vehicle_3d.visualization.utils import get_thrust_tf_ca

@dataclass
class MotorcycleTireConfig(PythonMsg):
    ''' configuration of motorcycle tire model '''

    Fz0: float = field(default = 2000)
    r: float = field(default = 0.3)
    Iyy: float = field(default = 0.3)

    d11: float = field(default = 14)
    d12: float = field(default = 13)
    d21: float = field(default = 9)
    d22: float = field(default = 4)
    d31: float = field(default = 0.8)
    d32: float = field(default = 0.8)
    d41: float = field(default = 1.2)
    d42: float = field(default = 1.2)
    d51: float = field(default = 0.15)
    d52: float = field(default = 0.4)
    d61: float = field(default = 0.1)
    d62: float = field(default = 0.1)
    d71: float = field(default = 0.15)
    d72: float = field(default = 0.15)
    d8: float = field(default = 1.6)

    e11: float = field(default = 0.4)
    e12: float = field(default = 0.4)
    e21: float = field(default = 0.04)
    e22: float = field(default = 0.07)
    e31: float = field(default = 0.08)
    e32: float = field(default = 0.1)
    e4: float = field(default = 10)
    e5: float = field(default = 2)
    e6: float = field(default = 1.5)
    e7: float = field(default = 50)
    e8: float = field(default = 1.1)
    e9: float = field(default = 20)
    e10: float = field(default = 1)

    f11: float = field(default = 1.5e-4)
    f12: float = field(default = 1.5e-4)
    f21: float = field(default = 1.0e-4)
    f22: float = field(default = 1.0e-4)

@dataclass
class MotorcycleTireState(VectorizablePythonMsg):
    ''' state of a single tire on the motorcycle '''
    N: float = field(default = 0.)
    ''' normal tire force '''
    Fx: float = field(default = 0.)
    ''' longitudinal tire force '''
    P: float = field(default = 0.)
    ''' driven wheel power '''
    y: float = field(default = 0.)
    ''' steering angle '''
    a: float = field(default = 0.)
    ''' slip angle '''
    c: float = field(default = 0.)
    ''' camber angle '''
    w: float = field(default = 0.)
    ''' wheel spin velocity '''

    def to_vec(self):
        return np.array([self.N, self.Fx, self.P, self.y,
            self.a, self.c, self.w])

    def from_vec(self, vec):
        self.N, self.Fx, self.P, self.y, \
            self.a, self.c, self.w = vec

@dataclass
class MotorcycleTire(PythonMsg):
    '''
    class to represent a tire that has been created
    stores net force and moment on the body frame as well as
    matrices for instanced rendering of tire forces
    '''
    a: ca.SX = field(default = None)
    y: ca.SX = field(default = None)
    c: ca.SX = field(default = None)
    w: ca.SX = field(default = None)
    Fx: ca.SX = field(default = None)
    Fb: ca.SX = field(default = None)
    Kb: ca.SX = field(default = None)
    P: ca.SX = field(default = None)
    normal_force_instance_matrix: ca.SX = field(default = None)
    tire_force_instance_matrix: ca.SX = field(default = None)

    def get_tire_state_vec(self):
        ''' return tire state vectors (MotorcycleTireState)'''
        return ca.vertcat(
            self.Fb[2],
            self.Fx,
            self.P,
            self.y,
            self.a,
            self.c,
            self.w,
        )

class MotorcycleTireModel:
    ''' model of motorcycle tire forces '''
    config: MotorcycleTireConfig

    def __init__(self, config: MotorcycleTireConfig):
        self.config = config

    def apply_model(self,
            x: ca.SX,
            x_com: ca.SX,
            vb: ca.SX,
            wb: ca.SX,
            Fz: ca.SX,
            Fx: ca.SX,
            y: ca.SX,
            c: ca.SX,
            rear: bool = True)\
        -> MotorcycleTire:
        '''
        apply tire model given
        x: position in body frame
        vb: body frame velocity
        wb: body frame angular velocity
        Fz: normal force on tire
        Fx: longitudinal tire force
        y: steering angle of tire
        c: camber angle of the tire
        rear: bool if this is the front or rear tire
        '''

        M = rotate(ca.DM_eye(4), y*180 / ca.pi, 0, 0, 1)
        M = translate(M, x[0], x[1], x[2])

        # tire frame velocity at tire contact patch
        vt = M[:3,:3].T @ (vb + hat(wb) @ x)
        # slip angle
        a = -ca.arctan(vt[1] / ca_pos_abs(vt[0]))
        # angular velocity estimate
        w = vt[0] / self.config.r

        if not rear:
            Fy, Mz = self.front_tire_force(
                Fx,
                Fz,
                c,
                a
            )
        else:
            Fy, Mz = self.rear_tire_force(
                Fx,
                Fz,
                c,
                a
            )

        Ft = ca.vertcat(Fx, Fy, Fz)
        Kt = ca.vertcat(0, 0, Mz)
        Fb = M[:3,:3] @ Ft
        Kb = hat(x_com) @ Fb + M[:3,:3] @ Kt

        # tire forces as global frame vectors
        Fx_b = M[:3,0] * Fx
        Fy_b = M[:3,1] * Fy
        Fz_b = M[:3,2] * Fz

        normal_force_instance_matrix = translate(ca.SX_eye(4), x[0], x[1], x[2])
        normal_force_instance_matrix[:3,:3] = get_thrust_tf_ca(Fz_b, norm=2000)
        Fx_instance_matrix = translate(ca.SX_eye(4), x[0], x[1], x[2])
        Fx_instance_matrix[:3,:3] = get_thrust_tf_ca(Fx_b, norm=2000)
        Fy_instance_matrix = translate(ca.SX_eye(4), x[0], x[1], x[2])
        Fy_instance_matrix[:3,:3] = get_thrust_tf_ca(Fy_b, norm=2000)
        tire_force_instance_matrix = ca.vertcat(
            Fx_instance_matrix,
            Fy_instance_matrix,
        )

        return MotorcycleTire(
            a = a,
            y = y,
            c = c,
            w = w,
            Fx = Fx,
            Fb = Fb,
            Kb = Kb,
            P = Fx * self.config.r * w,
            normal_force_instance_matrix=normal_force_instance_matrix,
            tire_force_instance_matrix=tire_force_instance_matrix,
        )

    def front_tire_force(self,
            Fx: ca.SX,
            Fz: ca.SX,
            c: ca.SX,
            a: ca.SX,
            ):
        ''' tire forces for the front wheel '''
        config = self.config
        CFa0 = config.d11 * config.Fz0 + config.d21 * (Fz - config.Fz0)
        CFa = CFa0 / (1 + config.d51*c**2)
        CFy = config.e11 * Fz
        C = config.d8
        K = CFa
        D0 = config.d41 * Fz / (1 + config.d71 * c**2)
        D = ca.sqrt(D0**2 - Fx**2)
        B = K / C / D0
        SHf = CFy * c / CFa
        SV = config.d61 * Fz * c * D / D0
        SH = SHf - SV / CFa
        a_eq = D0/D  *(a + SHf) - SHf
        Fy = D * ca.sin(C*ca.arctan(B*a_eq + SH)) + SV

        CMa = config.e11 * Fz
        CMy = config.e31 * Fz
        rc = config.e31
        ta0 = CMa / CFa0
        a_eq0 = D0/D*a
        Fya = D * ca.sin(C * ca.arctan(B*a_eq0))
        Bt = config.e7
        Ct = config.e8
        Br = config.e9 / (1 + config.e4 * c**2)
        Cr = config.e10 / (1 + config.e5*c**2)
        ta = ta0 * ca.cos(Ct*ca.arctan(Bt*a_eq0)) / (1 + config.e5 * c**2)
        Mzr0 = CMy * ca.arctan(config.e6 * c) / config.e6
        Mzr = Mzr0 * ca.cos(Cr * ca.arctan(Br*a_eq0))
        Mz = -ta* Fya + Mzr - rc*Fx*ca.tan(c)

        return Fy, Mz

    def rear_tire_force(self,
            Fx: ca.SX,
            Fz: ca.SX,
            c: ca.SX,
            a: ca.SX,
            ):
        ''' tire forces for the rear wheel '''
        config = self.config
        CFa0 = config.d12 * config.Fz0 + config.d22 * (Fz - config.Fz0)
        CFa = CFa0 / (1 + config.d52*c**2)
        CFy = config.e12 * Fz
        C = config.d8
        K = CFa
        D0 = config.d42 * Fz / (1 + config.d72 * c**2)
        D = ca.sqrt(D0**2 - Fx**2)
        B = K / C / D0
        SHf = CFy * c / CFa
        SV = config.d62 * Fz * c * D / D0
        SH = SHf - SV / CFa
        a_eq = D0/D  *(a + SHf) - SHf
        Fy_rear = D * ca.sin(C*ca.arctan(B*a_eq + SH)) + SV

        CMa = config.e12 * Fz
        CMy = config.e32 * Fz
        rc = config.e32
        ta0 = CMa / CFa0
        a_eq0 = D0/D*a
        Fya = D * ca.sin(C * ca.arctan(B*a_eq0))
        Bt = config.e7
        Ct = config.e8
        Br = config.e9 / (1 + config.e4 * c**2)
        Cr = config.e10 / (1 + config.e5*c**2)
        ta = ta0 * ca.cos(Ct*ca.arctan(Bt*a_eq0)) / (1 + config.e5 * c**2)
        Mzr0 = CMy * ca.arctan(config.e6 * c) / config.e6
        Mzr = Mzr0 * ca.cos(Cr * ca.arctan(Br*a_eq0))
        Mz_rear = -ta* Fya + Mzr - rc*Fx*ca.tan(c)
        return Fy_rear, Mz_rear
