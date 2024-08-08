'''
parametric surface defined by an elevation map over a Euclidean space
xp(s,y) = [s,y,h(s,y)]
'''
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import casadi as ca

from vehicle_3d.pytypes import BaseBodyState
from vehicle_3d.utils.ca_utils import check_free_sx
from vehicle_3d.utils.interp import RBF, RBF_KERNELS
from vehicle_3d.surfaces.base_surface import BaseSurfaceConfig, BaseSurface, \
    BaseSurfaceSymRep


@dataclass
class ElevationSurfaceConfig(BaseSurfaceConfig):
    ''' configuration needed for elevation surface'''
    s_min: float = field(default = -50.)
    s_max: float = field(default = 50.)
    y_min: float = field(default = -50.)
    y_max: float = field(default = 50.)
    closed: bool = field(default = False)

    # optional fields depending on type of instantiation
    # see subsequent classes
    sx_h: ca.Function = field(default = None)
    kernel: RBF_KERNELS  = field(default = RBF_KERNELS.GAUSSIAN)

@dataclass
class ElevationSurfaceFunctionConfig(ElevationSurfaceConfig):
    '''
    configuration format where the elevation map
    is an explicit function
    '''
    def __post_init__(self):
        if self.sx_h is None:
            s = ca.SX.sym('s')
            y = ca.SX.sym('y')
            h = ca.exp(-(s**2 + y**2)/500) * 20

            self.sx_h = h

@dataclass
class ElevationSurfaceInterpConfig(ElevationSurfaceConfig):
    '''
    configuration format where numpy arrays X,Y,Z
    are interpolated with a radial basis function
    '''
    X: np.ndarray = field(default = None)
    Y: np.ndarray = field(default = None)
    Z: np.ndarray = field(default = None)
    rbf_epsilon: float = field(default = 1/20)

    def __post_init__(self):
        if self.X is None or self.Y is None or self.Z is None:
            x = np.linspace(self.s_min, self.s_max, 5)
            y = np.linspace(self.y_min, self.y_max, 5)
            X, Y = np.meshgrid(x,y)
            Z = 20 * np.exp(-(X **2 + Y**2)/500 )
            self.X = X
            self.Y = Y
            self.Z = Z


class ElevationSurface(BaseSurface):
    '''
    An elevation map, otherwise known as a Mongue patch
    '''
    h: ca.Function = None
    dh = None
    ddh = None
    rbf: RBF = None

    def g2px(self, state: BaseBodyState, exact: bool = True):
        state.p.from_vec(state.x.to_vec())
        state.p.n -= float(self.h((state.x.xi, state.x.xj)))

        if exact:
            ubp = [self.s_max(), self.y_max(state.p.s), np.inf]
            lbp = [self.s_min(), self.y_min(state.p.s), -np.inf]
            state.p.from_vec(
                self.x2p(state.x.to_vec(), state.p.to_vec(), ubp, lbp)
            )

    def _setup_interp(self):
        if isinstance(self.config, ElevationSurfaceFunctionConfig):
            _, free_vars = check_free_sx([self.config.sx_h], [])
            sy = ca.vertcat(*free_vars)
            if sy.shape == (1,1):
                sy = ca.vertcat(sy, ca.SX.sym('y'))
            h = self.config.sx_h
            dh = ca.jacobian(h, sy)
            ddh = ca.jacobian(dh, sy)

            self.h = ca.Function('h', [sy], [h])
            self.dh = ca.Function('h', [sy], [dh])
            self.ddh = ca.Function('h', [sy], [ddh])

        elif isinstance(self.config, ElevationSurfaceInterpConfig):
            self.rbf = RBF(self.config.X, self.config.Y, self.config.Z,
                           epsilon = self.config.rbf_epsilon, kernel = self.config.kernel)

            self.h = self.rbf.get_rbf()
            self.dh = self.rbf.get_rbf_jac()
            self.ddh = self.rbf.get_rbf_hess()
        else:
            raise NotImplementedError('Unsupported config for fully initializing elevation surface')

    def _compute_sym_rep(self):
        self.sym_rep = BaseSurfaceSymRep()
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')

        # elevation and partial derivatives (symbolic)
        h = ca.SX.sym('h')
        hs = ca.SX.sym('hs')
        hy = ca.SX.sym('hy')
        hss = ca.SX.sym('hss')
        hsy = ca.SX.sym('hsy')
        hys = hsy
        hyy = ca.SX.sym('hyy')

        xp = ca.vertcat(s,y,h)

        # partial derivatives of parametric surface
        xps = ca.jacobian(xp, s) + ca.jacobian(xp,h) * hs
        xpy = ca.jacobian(xp, y) + ca.jacobian(xp,h) * hy

        xpss = ca.jacobian(xps, s) + \
               ca.jacobian(xps, ca.vertcat(h,hs,hy)) @ ca.vertcat(hs, hss, hys)
        xpsy = ca.jacobian(xps, y) + \
               ca.jacobian(xps, ca.vertcat(h,hs,hy)) @ ca.vertcat(hy, hsy, hyy)
        xpys = xpsy
        xpyy = ca.jacobian(xpy, y) + \
               ca.jacobian(xpy, ca.vertcat(h,hs,hy)) @ ca.vertcat(hy, hsy, hyy)

        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(h,hs,hy,hss,hsy,hyy)

        sy = ca.vertcat(s, y)
        param_terms_explicit = ca.vertcat(self.h(sy), self.dh(sy).T, self.ddh(sy)[0,1,3])
        f_param_terms = ca.Function('param_terms',[s, y, n], [param_terms_explicit])
        eval_param_terms = f_param_terms(s, y, 0)

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

        self.sym_rep.param_terms = param_terms
        self.sym_rep.f_param_terms = f_param_terms
        self.sym_rep.eval_param_terms = eval_param_terms

    def pose_eqns_2D_planar(self, vb1: ca.SX, vb2: ca.SX, wb3: ca.SX, n:float=0)\
            -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        ths = self.sym_rep.ths

        s_dot = vb1 * ca.cos(ths) - vb2 * ca.sin(ths)
        y_dot = vb1 * ca.sin(ths) + vb2 * ca.cos(ths)
        ths_dot = wb3
        w1 = 0
        w2 = 0
        return s_dot, y_dot, ths_dot, w1, w2
