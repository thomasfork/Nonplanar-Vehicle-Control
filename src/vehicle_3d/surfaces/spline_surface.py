'''
A surface parameterized by a 3D spline centerline

it is assumed that the spline never points in the vertical direction [0,0,1]
otherwise the lateral direction would not be well defined
'''
from typing import Tuple, Union, Callable
from dataclasses import dataclass, field

import numpy as np
import casadi as ca
import scipy.interpolate

from vehicle_3d.utils.interp import spline_interpolant
from vehicle_3d.surfaces.base_surface import BaseCenterlineSurfaceConfig, \
    BaseCenterlineSurfaceSymRep, BaseLinearCenterlineSurface

@dataclass
class SplineSurfaceConfig(BaseCenterlineSurfaceConfig):
    '''
    spline-based frenet surface config
    definining parameters are s and x, which are abscissa and position
    abscissa does not need to be arc length
    x should be of shape N x 3
    '''

    x: np.ndarray = field(default = None)
    y_max_interp: Union[None, np.ndarray] = field(default = None)
    y_min_interp: Union[None, np.ndarray] = field(default = None)

    def __post_init__(self):
        if self.x is None:
            self.x = np.array([[0,0,0],[10,0,0],[10,10,5],[0,10,0]])
            self.closed = True
        if self.s is None:
            self.s = np.arange(self.x.shape[0])

class SplineSurface(BaseLinearCenterlineSurface):
    ''' spline=based frenet frame surface'''
    xc: ca.Function
    dxc: ca.Function
    ddxc: ca.Function
    d3xc: ca.Function

    config: SplineSurfaceConfig

    _xc_interp: Callable[[np.ndarray], np.ndarray]
    _y_max_interp: Callable[[np.ndarray], np.ndarray] = None
    _y_min_interp: Callable[[np.ndarray], np.ndarray] = None

    def p2x_fast(self, s, y, n):
        xc = self._xc_interp(s.squeeze()).T
        xcs = self._xc_interp(s.squeeze(), 1).T
        xcp = np.vstack([-xcs[1], xcs[0], np.zeros((1, xc.shape[1]))])
        ey = xcp / np.linalg.norm(xcp, axis = 0)
        return xc + y * ey

    def p2xpn_fast(self, s, y):
        xc = self._xc_interp(s.squeeze()).T
        xcs = self._xc_interp(s.squeeze(), 1).T
        xcp = np.vstack([-xcs[1], xcs[0], np.zeros((1, xc.shape[1]))])
        ey = xcp / np.linalg.norm(xcp, axis = 0)
        es = xcs / np.linalg.norm(xcs, axis = 0)
        # slight approximation here
        return np.cross(es, ey, axisa = 0, axisb = 0, axisc = 0)

    def __init__(self, config: SplineSurfaceConfig):
        config.flat = np.all(config.x[:,2] == 0)
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        config.orthogonal = True
        super().__init__(config)

    def y_max(self, s = 0):
        if self._y_max_interp is not None:
            return self._y_max_interp(s)
        return super().y_max(s)

    def y_min(self, s = 0):
        if self._y_min_interp is not None:
            return self._y_min_interp(s)
        return super().y_min(s)

    def _setup_interp(self):
        if self.config.closed:
            if not (self.config.x[0] == self.config.x[-1]).all():
                self.config.x = np.vstack([self.config.x, self.config.x[0]])
                self.config.s = np.concatenate((self.config.s, (self.s_max()+1,)))
                self.config.s_max += 1

                if self.config.y_max_interp is not None:
                    self.config.y_max_interp = \
                        np.concatenate([self.config.y_max_interp, self.config.y_max_interp[0:1]])

                if self.config.y_min_interp is not None:
                    self.config.y_min_interp = \
                        np.concatenate([self.config.y_min_interp, self.config.y_min_interp[0:1]])

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

        # internal splines for fast computation
        self._xc_interp = scipy.interpolate.CubicSpline(
                self.config.s, self.config.x, bc_type = bc)
        if self.config.y_max_interp is not None:
            self._y_max_interp = scipy.interpolate.CubicSpline(
                self.config.s, self.config.y_max_interp, bc_type = bc)
        if self.config.y_min_interp is not None:
            self._y_min_interp = scipy.interpolate.CubicSpline(
                self.config.s, self.config.y_min_interp, bc_type = bc)

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

        esc = dxc / ca.norm_2(dxc)
        eyc = ca.vertcat(-esc[1], esc[0], 0)
        eyc = eyc / ca.norm_2(eyc)
        enc = ca.cross(esc, eyc)

        xp = xc + y * eyc

        xps = ca.jacobian(xp, xc) @ dxc \
            + ca.jacobian(xp, dxc) @ ddxc
        xpy = ca.jacobian(xp, y)

        xpss = ca.jacobian(xps, xc) @ dxc \
            + ca.jacobian(xps, dxc) @ ddxc \
            + ca.jacobian(xps, ddxc) @ d3xc
        xpsy = ca.jacobian(xps, y)
        xpys = xpsy
        xpyy = ca.jacobian(xpy, y)

        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(xc, dxc, ddxc, d3xc)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.xc(s),
            self.dxc(s),
            self.ddxc(s),
            self.d3xc(s))
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

    def triangulate_num_s(self) -> int:
        return max(1000, self.s_max() * 10)
