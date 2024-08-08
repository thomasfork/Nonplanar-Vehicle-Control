'''
A surface where the left and right boundaries are given with splines
and the surface is a line between the two splines.

Both splines are allowed to be 3D.

Enforces y in [0,1] with 0 corresponding to the right boundary
and 1 corresponding to the left boundary
'''
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import casadi as ca
import scipy.interpolate

from vehicle_3d.surfaces.base_surface import BaseCenterlineSurface, BaseCenterlineSurfaceConfig,\
    BaseCenterlineSurfaceSymRep
from vehicle_3d.utils.interp import spline_interpolant

@dataclass
class BoundarySplineSurfaceConfig(BaseCenterlineSurfaceConfig):
    '''
    configuration object for spline surface
    's' will never correspond to the exact path length, but will be close.
    This is because 's' is used as the parameterization variable for interpolation,
    which is usually slightly different from path length.

    setting closed = True is recommended for closed tracks; this will not impact the surface setup
    but is used in raceline computation code
    '''

    # centerline interpolation path length
    s:   np.ndarray = field(default = np.array([0,10,20,30]))
    # boundary interpolation position, two Nx3 arrays
    xl:   np.ndarray = field(default = np.array([[0,0,0],[10,0,0],[10,10,0],[0,10,0]]))
    xr:   np.ndarray = field(default = np.array([[0,-4,0],[14,-4,1],[14,14,1],[0,14,0]]))

    y_min: float = field(default = 0)
    y_max: float = field(default = 1)

    def __post_init__(self):
        self.y_max = 1
        self.y_min = 0

class BoundarySplineSurface(BaseCenterlineSurface):
    ''' Spline centerline surface, an instance of a centerline surface'''
    config: BoundarySplineSurfaceConfig

    xr = None
    dxr = None
    ddxr = None
    xl = None
    dxl = None
    ddxl = None

    # scipy splines for the boundaries
    # meant for generating surface textures fast
    _xr_spline = None
    _xl_spline = None

    def __init__(self, config: BoundarySplineSurfaceConfig):
        config.flat = np.all(config.xl[:,2] == 0) and np.all(config.xr[:,2] == 0)
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        config.orthogonal = False
        super().__init__(config)

    def p2x_fast(self, s, y, n):
        xl = self._xl_spline(s.squeeze()).T
        xr = self._xr_spline(s.squeeze()).T
        return xl * y + xr * (1 - y)

    def p2xpn_fast(self, s, y):
        xl = self._xl_spline(s.squeeze()).T
        xr = self._xr_spline(s.squeeze()).T
        xls = self._xl_spline(s.squeeze(), 1).T
        xrs = self._xr_spline(s.squeeze(), 1).T
        xps = xls * y.squeeze() + xrs * (1-y.squeeze())
        xpy = xl - xr
        xpn = np.cross(xps, xpy, axisa = 0, axisb = 0, axisc = 0)
        xpn = xpn / np.linalg.norm(xpn, axis = 0)
        return xpn

    def _setup_interp(self):
        assert isinstance(self.config, BoundarySplineSurfaceConfig)
        bc_type = 'not-a-knot' if not self.config.closed else 'periodic'
        self._xr_spline = \
            scipy.interpolate.CubicSpline(self.config.s, self.config.xr, bc_type = bc_type)
        self._xl_spline = \
            scipy.interpolate.CubicSpline(self.config.s, self.config.xl, bc_type = bc_type)

        xli = spline_interpolant(self.config.s, self.config.xl[:,0], extrapolate = 'linear',
            bc_type = bc_type)
        xlj = spline_interpolant(self.config.s, self.config.xl[:,1], extrapolate = 'linear',
            bc_type = bc_type)
        xlk = spline_interpolant(self.config.s, self.config.xl[:,2], extrapolate = 'linear',
            bc_type = bc_type)
        xri = spline_interpolant(self.config.s, self.config.xr[:,0], extrapolate = 'linear',
            bc_type = bc_type)
        xrj = spline_interpolant(self.config.s, self.config.xr[:,1], extrapolate = 'linear',
            bc_type = bc_type)
        xrk = spline_interpolant(self.config.s, self.config.xr[:,2], extrapolate = 'linear',
            bc_type = bc_type)

        s = ca.SX.sym('s')

        xl = ca.vertcat(xli(s), xlj(s), xlk(s))
        xr = ca.vertcat(xri(s), xrj(s), xrk(s))

        self.xl   = ca.Function('xl',   [s], [xl])
        self.dxl  = ca.Function('dxl',  [s], [ca.jacobian(self.xl(s), s)])
        self.ddxl = ca.Function('ddxl', [s], [ca.jacobian(self.dxl(s), s)])

        self.xr   = ca.Function('xr',   [s], [xr])
        self.dxr  = ca.Function('dxr',  [s], [ca.jacobian(self.xr(s), s)])
        self.ddxr = ca.Function('ddxr', [s], [ca.jacobian(self.dxr(s), s)])

    def _setup_centerline(self):
        # automatically set up by _setup_interp for this surface
        pass

    def _compute_sym_rep(self):
        self.sym_rep = BaseCenterlineSurfaceSymRep()
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')

        xl   = ca.SX.sym('xl',   3)
        dxl  = ca.SX.sym('dxl',  3)
        ddxl = ca.SX.sym('ddxl', 3)

        xr   = ca.SX.sym('xr',   3)
        dxr  = ca.SX.sym('dxr',  3)
        ddxr = ca.SX.sym('ddxr', 3)

        esc = dxr / ca.norm_2(dxr)
        eyc = xl - xr
        eyc = eyc / ca.norm_2(eyc)
        enc = ca.cross(esc, eyc)

        xp = xl * y + xr * (1-y)

        xps = ca.jacobian(xp, xl) @ dxl + \
              ca.jacobian(xp, xr) @ dxr
        xpy = ca.jacobian(xp, y)

        xpss = ca.jacobian(xps, xl) @ dxl + \
               ca.jacobian(xps, dxl) @ ddxl + \
               ca.jacobian(xps, xr) @ dxr + \
               ca.jacobian(xps, dxr) @ ddxr
        xpsy = ca.jacobian(xps, y)
        xpys = xpsy
        xpyy = ca.jacobian(xpy, y)

        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(xr, dxr, ddxr, xl, dxl, ddxl)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.xr(s),
            self.dxr(s),
            self.ddxr(s),
            self.xl(s),
            self.dxl(s),
            self.ddxl(s))
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

        self.sym_rep.xc = xr
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
