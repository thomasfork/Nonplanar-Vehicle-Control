'''
A surface parameterized by a 3D spline centerline
and a lateral bank angle

it is assumed that the spline never points in the vertical direction [0,0,1]
otherwise the bank angle rotation would not be well defined

bank angle is rotation about the tangent vector of the centerline after projection to 2D
This is slightly different than rotation about the centerline itself,
and is done to make it easier to obtain bank angle and spline data from elevation data,
such as for obtaining a frenet-like frame to follow a road given elevation data

positive bank angle corresponds to right handed rotation about the tangent direction
of the centerline spline
'''
from typing import Tuple
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from vehicle_3d.utils.interp import InterpMethod, interp, spline_interpolant
from vehicle_3d.surfaces.base_surface import BaseCenterlineSurfaceConfig, \
    BaseCenterlineSurfaceSymRep, BaseLinearCenterlineSurface

@dataclass
class BankedSplineSurfaceConfig(BaseCenterlineSurfaceConfig):
    '''
    spline-based frenet surface config
    definining parameters are s and x, which are abscissa and position
    abscissa does not need to be arc length
    x should be of shape N x 3
    b is the bank angle of the surface at each x waypoint
    '''

    x: np.ndarray = field(default = None)
    b: np.ndarray = field(default = None)
    b_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)

    def __post_init__(self):
        if self.x is None:
            self.x = np.array([[0,0,0],[10,0,0],[10,10,5],[0,10,0]])
            self.closed = True
        if self.s is None:
            self.s = np.arange(self.x.shape[0])
        if self.b is None:
            self.b = np.array([0,0.4,0.2,0])

class BankedSplineSurface(BaseLinearCenterlineSurface):
    ''' spline=based frenet frame surface'''
    xc: ca.Function
    dxc: ca.Function
    ddxc: ca.Function
    d3xc: ca.Function
    b: ca.Function
    db: ca.Function
    ddb: ca.Function

    config: BankedSplineSurfaceConfig

    def __init__(self, config: BankedSplineSurfaceConfig):
        config.flat = np.all(config.x[:,2] == 0) and np.all(config.b == 0)
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        config.orthogonal = True
        super().__init__(config)

    def _setup_interp(self):
        if self.config.closed:
            if not (self.config.x[0] == self.config.x[-1]).all():
                self.config.x = np.vstack([self.config.x, self.config.x[0]])
                self.config.s = np.concatenate((self.config.s, (self.s_max()+1,)))
                self.config.b = np.concatenate((self.config.b, (self.config.b[0],)))
                self.config.s_max += 1

        bc = 'periodic' if self.config.closed else 'not-a-knot'
        ext = 'linear'

        s = ca.SX.sym('s')

        xi = spline_interpolant(self.config.s, self.config.x[:,0],
                                bc_type=bc, extrapolate = ext)
        xj = spline_interpolant(self.config.s, self.config.x[:,1],
                                bc_type=bc, extrapolate = ext)
        if np.all(self.config.x[:,2] == 0):
            xk = ca.Function('xk',[s],[0])
        else:
            xk = spline_interpolant(self.config.s, self.config.x[:,2],
                                    bc_type=bc, extrapolate = ext)

        xc = ca.vertcat(xi(s), xj(s), xk(s))
        self.xc   = ca.Function('xc',   [s], [xc])
        self.dxc  = ca.Function('dxc',  [s], [ca.jacobian(self.xc(s), s)])
        self.ddxc = ca.Function('ddxc', [s], [ca.jacobian(self.dxc(s), s)])
        self.d3xc = ca.Function('d3xc', [s], [ca.jacobian(self.ddxc(s), s)])

        self.b = interp(self.config.s, self.config.b, self.config.b_interp_method,
                            extrapolate=ext, spline_bc=bc)
        self.db  = ca.Function('db',  [s], [ca.jacobian(self.b(s), s)])
        self.ddb  = ca.Function('ddb',  [s], [ca.jacobian(self.db(s), s)])

    def _setup_centerline(self):
        # automatically set up by _setup_interp for this surface
        pass

    def _compute_sym_rep(self):
        self.sym_rep = BaseCenterlineSurfaceSymRep()
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')

        xc   = ca.SX.sym('xc',   3)
        dxc  = ca.SX.sym('dxc',  3)
        ddxc = ca.SX.sym('ddxc', 3)
        d3xc = ca.SX.sym('d3xc', 3)

        b = ca.SX.sym('b')
        db = ca.SX.sym('db')
        ddb = ca.SX.sym('ddb')

        esc = dxc / ca.norm_2(dxc)
        e2c = ca.vertcat(-esc[1], esc[0], 0)
        e2c = e2c / ca.norm_2(e2c)
        e3c = ca.cross(esc, e2c)

        eyc = ca.cos(b) * e2c + ca.sin(b) * e3c
        enc = ca.cos(b) * e3c - ca.sin(b) * e2c

        xp = xc + y * eyc

        xps = ca.jacobian(xp, xc) @ dxc \
            + ca.jacobian(xp, dxc) @ ddxc \
            + ca.jacobian(xp, b) @ db
        xpy = ca.jacobian(xp, y)

        xpss = ca.jacobian(xps, xc) @ dxc \
            + ca.jacobian(xps, dxc) @ ddxc \
            + ca.jacobian(xps, ddxc) @ d3xc \
            + ca.jacobian(xp, b) @ db \
            + ca.jacobian(xp, db) @ ddb
        xpsy = ca.jacobian(xps, y)
        xpys = xpsy
        xpyy = ca.jacobian(xpy, y)

        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(xc, dxc, ddxc, d3xc, b, db, ddb)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.xc(s),
            self.dxc(s),
            self.ddxc(s),
            self.d3xc(s),
            self.b(s),
            self.db(s),
            self.ddb(s))
        f_param_terms = ca.Function('param_terms',[s, y, n], [param_terms_explicit])

        eval_param_terms = f_param_terms(s, 0, 0)

        self.sym_rep.s = s
        self.sym_rep.y = y
        self.sym_rep.n = n
        self.sym_rep.xp = xp
        self.sym_rep.xps = xps
        self.sym_rep.xpy = xpy
        self.sym_rep.xpss = xpss
        self.sym_rep.xpsy = xpsy
        self.sym_rep.xpys = xpys
        self.sym_rep.xpyy = xpyy

        self.sym_rep.xc = xc
        self.sym_rep.ecs = esc
        self.sym_rep.ecy = eyc
        self.sym_rep.ecn = enc

        self.sym_rep.param_terms = param_terms
        self.sym_rep.f_param_terms = f_param_terms
        self.sym_rep.eval_param_terms = eval_param_terms

    def pose_eqns_2D_planar(self, vb1: ca.SX, vb2: ca.SX, wb3: ca.SX, n:float=0)\
            -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        y = self.sym_rep.y
        ths = self.sym_rep.ths
        xps_mag = self.p2mag_xps(self.sym_rep.s,0)
        kn = - ca.cross(self.sym_rep.xpss, self.sym_rep.xps).T @ \
            ca.vertcat(0,0,1) / ca.norm_2(self.sym_rep.xps)**3
        kappa = ca.substitute(kn, self.sym_rep.y, 0)

        s_dot = (vb1 * ca.cos(ths) - vb2 * ca.sin(ths)) / (1- kappa * y) / xps_mag
        y_dot =  vb1 * ca.sin(ths) + vb2 * ca.cos(ths)
        ths_dot = wb3 - kappa * s_dot * xps_mag
        w1 = 0
        w2 = 0
        return s_dot, y_dot, ths_dot, w1, w2
