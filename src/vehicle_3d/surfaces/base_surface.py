'''
standard format for surface parameterization classes
'''

from abc import abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Callable, Union, Tuple

import casadi as ca
import numpy as np
from scipy.spatial import KDTree

try:
    import matplotlib.pyplot as plt
    _PYPLOT_AVAILABLE = True
except ImportError:
    _PYPLOT_AVAILABLE = False

from vehicle_3d.pytypes import PythonMsg, BaseBodyState, BaseTangentBodyState, Domain
from vehicle_3d.utils.ca_utils import ca_function_dispatcher, check_free_sx
from vehicle_3d.utils.ipopt_utils import ipopt_solver
from vehicle_3d.utils.rotations import Rotation, Reference
from vehicle_3d.visualization.shaders import vtype
from vehicle_3d.visualization.utils import join_vis
from vehicle_3d.visualization.objects import VertexObject, UBOObject
from vehicle_3d.visualization.window import Window

@dataclass
class BaseSurfaceConfig(PythonMsg):
    '''
    configuration parameters for surface
    '''
    s_max: float = field(default = 1.)
    ''' maximum s coordinate of the surface parameterization '''
    s_min: float = field(default = 0.)
    ''' minimum s coordinate of the surface parameterization '''
    y_max: float = field(default = 2.)
    ''' maximum y coordinate of the surface parameterization '''
    y_min: float = field(default = -2.)
    ''' minimum y coordinate of the surface parameterization '''
    closed: bool = field(default = False)
    ''' a closed surface will be periodic about max/min s bounds '''
    flat: bool = field(default = False)
    ''' flag set by surface class if surface is flat '''
    orthogonal: bool = field(default = False)
    ''' flag set by surface class if surface parameterization is orthogonal '''
    mx_xp: bool = field(default = False)
    ''' flag set by surface class if surface shape is implicit and requires ca.MX evaluation '''
    y_invariant: bool = field(default = False)
    ''' flag set by surface class if parameterization terms vary only with respect to s '''

@dataclass
class BaseSurfaceSymRep(PythonMsg):
    '''
    Symbolic representation of surface
    Expressions are functions of position on the surface (s,y) and sometimes (n)
    All expressions are functions of surface class parameterization terms (param_terms)
    '''

    s: ca.SX = field(default = None)
    ''' first parametric surface coordinate '''
    y: ca.SX = field(default = None)
    ''' second parametric surface coordinate '''
    n: ca.SX = field(default = None)
    ''' surface normal coordinate '''
    xp: Union[ca.SX, ca.MX] = field(default = None)
    ''' xp(s,y), the shape of the surface '''
    xps: ca.SX = field(default = None)
    ''' symbolic partial derivative of surface with respect to s '''
    xpy: ca.SX = field(default = None)
    ''' symbolic partial derivative of surface with respect to y '''
    xpss: ca.SX = field(default = None)
    ''' symbolic second partial of surface w.r.t s-s '''
    xpsy: ca.SX = field(default = None)
    ''' symbolic second partial of surface w.r.t s-y '''
    xpys: ca.SX = field(default = None)
    ''' symbolic second partial of surface w.r.t y-s '''
    xpyy: ca.SX = field(default = None)
    ''' symbolic second partial of surface w.r.t y-y '''

    # parameterization terms
    param_terms: ca.SX = field(default = None)
    ''' symbolic parameterization terms of the surface class '''
    f_param_terms: ca.Function = field(default = None)
    ''' a function from (s,y) to parameterization terms '''
    eval_param_terms: ca.SX = field(default = None)
    ''' parameterization terms of the surface in terms of (s, y) '''
    param_dim: float = field(default = None)
    ''' dimension of parameterization terms '''

    # derived terms
    p: ca.SX = field(default = None)
    ''' (s,y,n) '''
    xpn: ca.SX = field(default = None)
    ''' symbolic normal vector of the surface '''
    x: Union[ca.SX, ca.MX] = field(default = None)
    ''' symbolic position xp + n * xpn '''
    thp: ca.SX = field(default = None)
    ''' anglular measure of how far xps and xpy are from orthogonal '''
    eps: ca.SX = field(default = None)
    ''' symbolic normalization of xps '''
    epp: ca.SX = field(default = None)
    ''' symbolic orthonormalization of xpy w.r.t xps '''
    epn: ca.SX = field(default = None)
    ''' symbolic normal vector of the surface '''
    Rp: ca.SX = field(default = None)
    ''' orientation matrix [eps, epp, epn] '''
    one: ca.SX = field(default = None)
    ''' symbolic first fundamental form of the surface '''
    two: ca.SX = field(default = None)
    ''' symbolic second fundamental form of the surface '''
    ws: ca.SX = field(default = None)
    ''' angular rate coefficient from rate of change of s '''
    wy: ca.SX = field(default = None)
    ''' angular rate coefficient from rate of change of y '''
    I3: ca.SX = field(default = None)
    ''' symbolic metric tensor of x '''
    Q: ca.SX = field(default = None)
    ''' Q term of Q-R decomposition of J '''
    Q_inv: ca.SX = field(default = None)
    ''' inverse of Q '''

    # special terms for tangent contact
    ths: ca.SX = field(default = None)
    ''' angle between xps and longitudinal direction of a vehicle '''
    R_ths: ca.SX = field(default = None)
    ''' orientation matrix of vehicle '''
    J: ca.SX = field(default = None)
    ''' jacobian between vehicle and surface '''
    J_inv: ca.SX = field(default = None)
    ''' inverse of J '''

    # optional terms for surfaces that need mx evaluation for xp and x
    p_mx: ca.MX = field(default = None)
    ''' symbolic ca.MX version of p, populated when xp and x are ca.MX '''


class BaseSurface(Domain):
    ''' base surface class '''
    config: BaseSurfaceConfig
    ''' configuration of the surface '''
    sym_rep: BaseSurfaceSymRep
    ''' symbolic variables and expressions of the surface '''

    p2x: Callable[[float, float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y,n) -> 3D position '''
    p2xp: Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> 3D position on surface '''
    p2xps: Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> s tangent vector of surface '''
    p2mag_xps: Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> Euclidean norm of s tangent vector '''
    p2xpy: Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> y tangent vector of surface '''
    p2eps: Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> s tangent unit vector of surface '''
    p2epp: Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> perpendicular tangent unit vector of surface '''
    p2xpn: Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> normal vector of surface '''
    p2Rp:   Callable[[float, float], Union[ca.SX, np.ndarray]]
    ''' (s,y) -> [eps, epy, epn] orientation matrix'''

    pths2Rths: Callable[[float, float, float], Union[ca.SX, np.ndarray]]
    ''' converter for (s,y,ths) -> R for tangent contact '''

    x2p: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    '''
    optimization-based converter for global to parametric position
    arguments:
        0: (xi,xj,xk) known 3D Euclidean position
        1: (s,y,n)    initial guess for parametric position
        2: (p_max)    upper bounds on (s,y,n)
        3: (p_min)    lower bounds on (s,y,n)
    returns:
        (s,y,n)       locally optimized parametric position
    '''

    def __init__(self, config: BaseSurfaceConfig):
        self.config = config
        self._setup_interp()
        self._compute_sym_rep()
        self._post_sym_rep()
        self._check_sym_rep()
        self._setup_helper_functions()
        self.periodic = self.config.closed

    def s_max(self, s: Union[float, np.ndarray] = 0) -> float:
        ''' maximum s coordinate, always constant '''
        if isinstance(s, np.ndarray):
            return self.config.s_max * np.ones(s.shape)
        return self.config.s_max

    def s_min(self, s: Union[float, np.ndarray] = 0) -> float:
        ''' minimum s coordinate, always constant '''
        if isinstance(s, np.ndarray):
            return self.config.s_min * np.ones(s.shape)
        return self.config.s_min

    def y_max(self, s: Union[float, np.ndarray] = 0) -> float:
        ''' maximum y coordinate, may be variable with s for some surfaces '''
        if isinstance(s, np.ndarray):
            return self.config.y_max * np.ones(s.shape)
        return self.config.y_max

    def y_min(self, s: Union[float, np.ndarray] = 0) -> float:
        ''' minimum y coordinate, may be variable with s for some surfaces '''
        if isinstance(s, np.ndarray):
            return self.config.y_min * np.ones(s.shape)
        return self.config.y_min

    def fill_in_param_terms(self,
        expr: Union[ca.SX, ca.MX],
        args: Tuple[Union[ca.SX, ca.MX]],
        dispatch: bool = True):
        '''
        helper for obtaining surface-dependent functions
        makes parameterization implicit provided that dependent variables
        are included in args.

        Example:
            (y) depends on (x) and surface parameterization terms (surf.sym_rep.param_terms)
            (x) must include any variables that determine (surf.sym_rep.param_terms)
                (s) for any surface that isn't Euclidean
                (y) for any surface with surf.config.y_invariant == False
            surf.fill_in_param_terms(y, x) 
                returns a function f(x) -> y 
                dependence on param_terms has been made implicit
                otherwise one would need to create a function
                f(x, param_terms) -> y
        '''

        if not isinstance(expr, list):
            expr = [expr]
        if not isinstance(args, list):
            args = [args]
        if isinstance(args[0], ca.SX):
            func, _ = check_free_sx(expr, args)
            if func is None:
                gen_func = ca.Function('f', [*args, self.sym_rep.param_terms], expr)
                func_eval = gen_func(*args, self.sym_rep.eval_param_terms)

                func, free_vars = check_free_sx([func_eval], args)
                if func is None:
                    raise RuntimeError(
                        f'Invalid Function Created, {free_vars} are not available')
        else:
            # MX expressions cannot be derived in a manner that requires above subs.
            func = ca.Function('f', args, expr)
            if len(func.free_mx()) > 0:
                raise RuntimeError(f'Invalid Function Created, {func.free_mx()} are not available')

        if dispatch:
            return ca_function_dispatcher(func)
        return func

    @abstractmethod
    def _setup_interp(self):
        ''' set up interpolation objects for the surface '''

    @abstractmethod
    def _compute_sym_rep(self):
        ''' compute symbolic representation of the surface '''

    def _post_sym_rep(self):
        ''' compute terms derived from symbolic representation of the surface '''
        self.sym_rep.param_dim = self.sym_rep.param_terms.shape[0]
        s = self.sym_rep.s
        y = self.sym_rep.y
        n = self.sym_rep.n
        xp = self.sym_rep.xp
        xps = self.sym_rep.xps
        xpy = self.sym_rep.xpy
        xpss = self.sym_rep.xpss
        xpsy = self.sym_rep.xpsy
        xpys = self.sym_rep.xpys
        xpyy = self.sym_rep.xpyy

        # pose
        p = ca.vertcat(s, y, n)

        # normal vector
        if self.config.flat:
            xpn = ca.DM((0,0,1))
        else:
            xpn = ca.cross(xps, xpy)
            xpn = xpn / ca.norm_2(xpn)

        # full 3D position
        if self.config.mx_xp:
            assert self.sym_rep.p_mx is not None
            p_mx = self.sym_rep.p_mx

            norm_offset = self.fill_in_param_terms(n * xpn, [p])(p_mx)
            x = xp + norm_offset

        else:
            x = xp + n * xpn

        # angle between xps and xpy
        if self.config.orthogonal:
            thp = 0
        else:
            thp = - ca.arcsin(xps.T @ xpy / ca.norm_2(xps) / ca.norm_2(xpy))

        # orthonormalized vectors
        eps = xps / ca.norm_2(xps)
        xpp = xpy - eps * (xpy.T @ eps)
        epp = xpp / ca.norm_2(xpp)
        epn = xpn
        Rp = ca.horzcat(eps, epp, epn)

        # first fundamental form of parametric surface
        if self.config.orthogonal:
            one = ca.diag(ca.vertcat(xps.T @ xps, xpy.T @ xpy))
        else:
            one = ca.vertcat(
                ca.horzcat(xps.T @ xps, xps.T @ xpy),
                ca.horzcat(xps.T @ xpy, xpy.T @ xpy)
            )

        # second fundamental form of parametric surface
        if self.config.flat:
            two = ca.DM([[0,0],[0,0]])
        else:
            two = ca.vertcat(
                ca.horzcat(xpss.T @ xpn, xpsy.T @ xpn),
                ca.horzcat(xpys.T @ xpn, xpyy.T @ xpn)
            )

        # coefs for angular velocity about xpn from s and y change
        ws = (ca.cross(xpss, xps).T @ xpn) / (xps.T @ xps)
        wy = (ca.cross(xpsy, xps).T @ xpn) / (xps.T @ xps)

        # 3D metric tensor
        I3 = ca.vertcat(
            ca.horzcat(
                one - two * n,
                ca.vertcat(0, 0)
            ),
            ca.horzcat(0, 0, 1)
        )

        # QR factorization of J
        Q = ca.vertcat(
            ca.horzcat(ca.norm_2(xps), 0 ),
            ca.horzcat(-ca.sin(thp)*ca.norm_2(xpy), ca.cos(thp) * ca.norm_2(xpy))
        )
        Q_inv = ca.vertcat(
            ca.horzcat(1 / ca.norm_2(xps), 0),
            ca.horzcat(ca.tan(thp) / ca.norm_2(xps), 1 / ca.norm_2(xpy) / ca.cos(thp) )
        )

        self.sym_rep.p = p
        self.sym_rep.xpn = xpn
        self.sym_rep.x = x
        self.sym_rep.thp = thp
        self.sym_rep.eps = eps
        self.sym_rep.epp = epp
        self.sym_rep.epn = epn
        self.sym_rep.Rp = Rp
        self.sym_rep.one = one
        self.sym_rep.two = two
        self.sym_rep.ws = ws
        self.sym_rep.wy = wy
        self.sym_rep.I3 = I3
        self.sym_rep.Q_inv = Q_inv
        self.sym_rep.Q = Q

        # additional fields for tangent contact support
        ths = ca.SX.sym('ths')

        J = ca.vertcat(ca.horzcat(ca.cos(ths)       * ca.sqrt(xps.T @ xps),
                                 -ca.sin(ths)       * ca.sqrt(xps.T @ xps)),
                       ca.horzcat(ca.sin(ths - thp) * ca.sqrt(xpy.T @ xpy),
                                  ca.cos(ths - thp) * ca.sqrt(xpy.T @ xpy)))
        J_inv = ca.vertcat(ca.horzcat(ca.cos(ths), ca.sin(ths)),
                           ca.horzcat(-ca.sin(ths), ca.cos(ths))) \
            @ Q_inv
        R = ca.horzcat(ca.horzcat(xps, xpy) @ (ca.inv(one)  @ J), xpn)
        self.sym_rep.ths = ths
        self.sym_rep.J = J
        self.sym_rep.J_inv = J_inv
        self.sym_rep.R_ths = R

        if hasattr(ca, 'cse'):
            for name, expr in asdict(self.sym_rep).items():
                if isinstance(expr, ca.SX):
                    setattr(self.sym_rep, name, ca.cse(expr))

    def _check_sym_rep(self):
        for label, attr in asdict(self.sym_rep).items():
            if label in ['p_mx'] and not self.config.mx_xp:
                pass
            elif attr is None:
                raise RuntimeError(f'Incomplete surface setup, attribute {label} is NoneType')

    def _setup_helper_functions(self):
        '''
        set up helper functions from
        symbolic representation and interpolation objects
        '''
        s = self.sym_rep.s
        y = self.sym_rep.y
        n = self.sym_rep.n
        ths = self.sym_rep.ths
        if not self.config.mx_xp:
            self.p2x  = self.fill_in_param_terms(self.sym_rep.x,  [s,y,n])
            self.p2xp = self.fill_in_param_terms(self.sym_rep.xp, [s,y])
        else:
            s_mx, y_mx, n_mx = self.sym_rep.p_mx[0], self.sym_rep.p_mx[1],self.sym_rep.p_mx[2]
            self.p2x  = self.fill_in_param_terms(self.sym_rep.x,  [s_mx,y_mx,n_mx])
            self.p2xp = self.fill_in_param_terms(self.sym_rep.xp, [s_mx,y_mx])
        self.p2xps = self.fill_in_param_terms(self.sym_rep.xps, [s,y])
        self.p2mag_xps = self.fill_in_param_terms(ca.norm_2(self.sym_rep.xps), [s,y])
        self.p2xpy = self.fill_in_param_terms(self.sym_rep.xpy, [s,y])
        self.p2eps = self.fill_in_param_terms(self.sym_rep.eps, [s,y])
        self.p2epp = self.fill_in_param_terms(self.sym_rep.epp, [s,y])
        self.p2xpn = self.fill_in_param_terms(self.sym_rep.xpn, [s,y])
        self.p2Rp = self.fill_in_param_terms(self.sym_rep.Rp, [s,y])
        self.pths2Rths = self.fill_in_param_terms(self.sym_rep.R_ths, [s,y,ths])

        # create optimization-based high accuracy global-parametric position
        # converter
        if not self.config.mx_xp:
            x = ca.SX.sym('x', 3)
            delta_x = x - self.p2x(s,y,n)
            p = self.sym_rep.p
        else:
            x = ca.MX.sym('x', 3)
            delta_x = x - self.p2x(s_mx, y_mx, n_mx)
            p = self.sym_rep.p_mx

        prob = {
            'f': ca.bilin(np.eye(3), delta_x, delta_x),
            'x': p,
            'p': x
        }
        solver = ipopt_solver(prob)

        x0 = ca.MX.sym('x0', 3)
        p0 = ca.MX.sym('p0', 3)
        ubp = ca.MX.sym('ubp', 3)
        lbp = ca.MX.sym('lbp', 3)

        x2p = ca.Function(
            'p2x',
            [x0, p0, ubp, lbp],
            [solver(x0 = p0, p = x0, ubx = ubp, lbx = lbp)['x']])
        self.x2p = lambda x, p0, ubp, lbp: np.array(x2p(x, p0, ubp, lbp)).squeeze()

    def p2x_fast(self, s: np.ndarray, y: np.ndarray, n: np.ndarray) -> np.ndarray:
        '''
        function to convert parametric to global position
        which may be faster than surface.p2x depending on implementation
        '''
        return self.p2x(s, y, n)

    def p2xpn_fast(self, s: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        function to convert parametric position to normal vector
        which may be faster than surface.p2x depending on implementation
        '''
        return self.p2xpn(s, y)

    def wrap_s(self, state: BaseBodyState) -> None:
        ''' check if a states p.s field should be wrapped for closed surfaces '''
        if self.config.closed:
            if state.p.s > self.s_max():
                state.p.s -= (self.s_max() - self.s_min())
            elif state.p.s < self.s_min():
                state.p.s += (self.s_max() - self.s_min())

    def p2gx(self, state: BaseBodyState):
        ''' parametric to global position on a state structure '''
        state.x.from_vec(self.p2x(*state.p.to_vec()))

    def p2gq(self, state: BaseTangentBodyState):
        ''' parametric to global orientation on a state structure '''
        state.q.from_mat(self.pths2Rths(state.p.s, state.p.y, state.ths))

    @abstractmethod
    def g2px(self, state: BaseBodyState, exact: bool = True):
        ''' global to parametric position on a state structure '''

    def g2pq(self, state: BaseTangentBodyState, preserve_yaw: bool = False):
        '''
        global to parametric orientation on a state structure

        if preserve_yaw is true, this will keep eb1 @ [-sin(yaw), cos(yaw),0] = 0
        this is intended for state.q.from_yaw(yaw), followed by this call.

        otherwise ths is set to the angle between xps and eb1
        '''
        if preserve_yaw:
            state.ths = -float(np.arctan2(
                state.q.e2() @ self.p2eps(state.p.s, state.p.y),
                state.q.e2() @ self.p2epp(state.p.s, state.p.y)
            ))
        else:
            state.ths = float(np.arctan2(
                -state.q.e2() @ self.p2xps(state.p.s, state.p.y),
                state.q.e1() @ self.p2xps(state.p.s, state.p.y)
            ))

    def triangulate_num_s(self) -> int:
        ''' number of intervals to triangulate along s coordinate '''
        return 100

    def triangulate_num_y(self) -> int:
        ''' number of intervals to triangulate along y coordinate '''
        return 100

    def camera_follow_mat(self, s: float = 0., y: float = 0.):
        return self.p2Rp(s,y).T

    def triangulate(self, ubo: UBOObject) -> VertexObject:
        '''
        return a mesh surface for OpenGL rendering
        '''
        thickness = min(1.0, (self.y_max(0) - self.y_min(0)) / 10)
        n_s = self.triangulate_num_s()
        n_y = self.triangulate_num_y()
        s = np.linspace(self.s_min(), self.s_max(), n_s)
        Y = np.linspace(self.y_min(s), self.y_max(s), n_y).T
        S = np.repeat(s[:,np.newaxis], n_y, axis = 1)

        s = np.concatenate(S)
        y = np.concatenate(Y)

        V_top = self.p2x_fast(s[None], y[None], 0).T
        N_top = self.p2xpn_fast(s[None], y[None]).T
        V_bot = V_top - N_top * thickness
        N_bot = -N_top.copy()

        # center and length scale for the surface
        x_max = V_top.max(axis = 0)
        x_min = V_top.min(axis = 0)

        self.view_center = (x_max + x_min) / 2
        self.view_scale   = np.linalg.norm(x_max - x_min)

        # spread out border normals for visual effect
        idxs = y == self.y_max(s)
        N_bot[idxs] += self.p2xpy(s[idxs][None], y[idxs][None]).T
        idxs = y == self.y_min(s)
        N_bot[idxs] -= self.p2xpy(s[idxs][None], y[idxs][None]).T

        idxs = s == self.s_min()
        N_bot[idxs] += self.p2xps(s[idxs][None], y[idxs][None]).T
        idxs = s == self.s_max()
        N_bot[idxs] -= self.p2xps(s[idxs][None], y[idxs][None]).T

        if isinstance(self, BaseCenterlineSurface):
            # single lane color scheme
            c = - (y - self.y_max(s))*(y -self.y_min(s)) \
                / (self.y_max(s) - self.y_min(s))**2 * 2
            c = c.clip(0,1).squeeze()
        else:
            # smooth checkerboard
            p = max((self.y_max(s) - self.y_min(s)).max(), self.s_max() - self.s_min())
            c = np.cos(s * np.pi * 7 / p) * np.cos(y * np.pi * 7 / p)
            c = c - c.min()
            c = c / c.max() / 1.5

        C_top = np.array([c,c,c, np.ones(c.shape)]).T
        C_bot = np.concatenate([np.zeros((c.shape[0], 3)), np.ones((c.shape[0],1))], axis = 1)

        # now compute indices that correspond to triangles and edges for upper surface
        I_array = np.zeros((n_s-1, n_y-1, 2,3))

        idxs = np.arange(len(V_top)).reshape(n_s,n_y)

        I_array[:,:,0,0] = idxs[:-1,:-1]       # bot left
        I_array[:,:,0,1] = idxs[1: ,:-1]       # bot right
        I_array[:,:,0,2] = idxs[:-1,1: ]       # top left
        I_array[:,:,1,0] = idxs[1: ,1: ]       # top right
        I_array[:,:,1,1] = idxs[:-1,1: ]       # top left
        I_array[:,:,1,2] = idxs[1: ,:-1]       # bot right

        # merge lower and upper surfaces
        offset = V_top.shape[0]                     #index offset

        # copy for lower indices and reverse triangle order for face culling
        I_array_lower = I_array + offset
        I_array_lower[:,:,:,[0,1,2]] = I_array_lower[:,:,:,[0,2,1]]
        I_array_lower = I_array_lower.reshape(-1,3)
        I_array = I_array.reshape(-1,3)

        #combine the surfaces
        V_tot = np.vstack([V_top, V_bot])
        C_tot = np.vstack([C_top, C_bot])
        n_tot = np.vstack([N_top, N_bot])
        I_tot = np.vstack([I_array, I_array_lower])

        # add triangles along the sides of the track
        I_s_border = np.zeros((n_s-1, 4, 3))
        I_y_border = np.zeros((n_y-1, 4, 3))

        I_s_border[:, 0, 0] = idxs[:-1, 0]           # right  top start
        I_s_border[:, 0, 1] = idxs[:-1, 0] + offset  # right  bot start
        I_s_border[:, 0, 2] = idxs[1:,  0]           # right  top end
        I_s_border[:, 1, 0] = idxs[1:,  0] + offset  # right  bot end
        I_s_border[:, 1, 2] = idxs[:-1, 0] + offset  # right  bot start
        I_s_border[:, 1, 1] = idxs[1:,  0]           # right  top end
        I_s_border[:, 2, 0] = idxs[:-1,-1]           # left top start
        I_s_border[:, 2, 2] = idxs[:-1,-1] + offset  # left bot start
        I_s_border[:, 2, 1] = idxs[1:, -1]           # left top end
        I_s_border[:, 3, 0] = idxs[1:, -1] + offset  # left bot end
        I_s_border[:, 3, 1] = idxs[:-1,-1] + offset  # left bot start
        I_s_border[:, 3, 2] = idxs[1:, -1]           # left top end

        I_y_border[:, 0, 0] = idxs[0, :-1]           # left  top start
        I_y_border[:, 0, 2] = idxs[0, :-1] + offset  # left  bot start
        I_y_border[:, 0, 1] = idxs[0, 1: ]           # left  top end
        I_y_border[:, 1, 0] = idxs[0, 1: ] + offset  # left  bot end
        I_y_border[:, 1, 1] = idxs[0, :-1] + offset  # left  bot start
        I_y_border[:, 1, 2] = idxs[0, 1: ]           # left  top end
        I_y_border[:, 2, 0] = idxs[-1,:-1]           # right top start
        I_y_border[:, 2 ,1] = idxs[-1,:-1] + offset  # right bot start
        I_y_border[:, 2 ,2] = idxs[-1,1: ]           # right top end
        I_y_border[:, 3, 0] = idxs[-1,1: ] + offset  # right bot end
        I_y_border[:, 3, 2] = idxs[-1,:-1] + offset  # right bot start
        I_y_border[:, 3, 1] = idxs[-1,1: ]           # right top end


        I_tot = np.vstack([I_tot, I_s_border.reshape(-1,3), I_y_border.reshape(-1,3)])
        I_tot = np.concatenate(I_tot).astype(np.uint32)

        V = np.zeros(V_tot.shape[0], dtype = vtype)
        V['a_position'] = V_tot
        V['a_color']    = C_tot
        V['a_normal']   = n_tot

        if self.config.closed:
            V, I_tot = join_vis(((V, I_tot), self._triangulate_finishline()))

        if isinstance(self, BaseCenterlineSurface):
            V, I_tot = join_vis(((V, I_tot), self._triangulate_siderails(thickness)))

        return VertexObject(ubo, V, I_tot)

    def _triangulate_finishline(self):
        ''' generate vertex/index data for adding a finishline '''
        N = 20 # must be even
        s = np.ones(N) * self.s_max()
        y = np.linspace(self.y_min(self.s_max()), self.y_max(self.s_max()), N)

        es_array = self.p2xps(s[None], y[None])
        es_array = es_array / np.linalg.norm(es_array, axis = 0)
        en_array = self.p2xpn(s[None], y[None])

        V_array1 = self.p2x(s[None], y[None],
                            (self.y_max(self.s_max()) - self.y_min(self.s_max()))/1000)
        dy = np.linalg.norm(V_array1[:,1] - V_array1[:,0])
        V_array1 += es_array * dy
        V_array2 = V_array1 - es_array * dy
        V_array3 = V_array2 - es_array * dy

        I_array = np.zeros((N-1,2,2,3))

        # first strip
        I_array[:,0,0,0] = np.arange(N-1)
        I_array[:,0,0,1] = np.arange(1, N)
        I_array[:,0,0,2] = np.arange(N, 2*N-1)
        I_array[:,0,1,0] = np.arange(N+1, 2*N)
        I_array[:,0,1,1] = np.arange(N, 2*N-1)
        I_array[:,0,1,2] = np.arange(1, N)

        # second strip
        I_array[:,1] = I_array[:,0] + N

        V_array = np.concatenate([V_array1, V_array2, V_array3], axis = 1)
        n_array = np.concatenate([en_array, en_array, en_array], axis = 1)
        I_array = I_array.reshape((-1,3)).astype(np.uint32)

        # figure out which triangles are for black/white surfaces
        index_test = I_array[:,0]
        indicator = np.zeros(index_test.shape)
        indicator[::2] = np.mod(np.mod(index_test[::2], N) + np.floor_divide(index_test[::2], N), 2)
        indicator[1::2] = indicator[::2]

        indicator = indicator > 0.5

        V_array = V_array[:, I_array]
        n_array = n_array[:, I_array]

        C_array = np.zeros((4, len(I_array), 3))
        C_array[3] = 1
        C_array[:, indicator] = np.array([1,1,1,1])[:, np.newaxis, np.newaxis]

        V_array = V_array.reshape((3, -1,1)).squeeze()
        n_array = n_array.reshape((3, -1,1)).squeeze()
        C_array = C_array.reshape((4, -1,1)).squeeze()

        V = np.zeros(12 * (N-1), dtype=vtype)
        V['a_position'] = V_array.T
        V['a_normal'] = n_array.T
        V['a_color'] = C_array.T

        I = np.arange(len(V), dtype = np.uint32)

        return V, I

    def _triangulate_siderails(self, thickness: float):
        ''' generate vertex/index data for extending surface sides slightly'''
        s = np.linspace(self.s_min(), self.s_max(), self.triangulate_num_s())
        y_ext = thickness
        n = len(s)
        V, I = [], []
        for k, Y in enumerate([self.y_min(s), self.y_max(s)]):
            S = s

            ey_array = self.p2xpy(S[None], Y[None])
            ey_array = ey_array / np.linalg.norm(ey_array, axis = 0)

            V_array1 = self.p2x(S[None], Y[None], 0)
            if k == 1:
                V_array1 += ey_array * y_ext
            V_array2 = self.p2x(S[None], Y[None], - thickness)
            if k == 1:
                V_array2 += ey_array * y_ext

            n_array1 = self.p2xpn(S[None], Y[None])
            n_array2 = -.1 * n_array1
            if k == 1:
                n_array2 += ey_array
            else:
                n_array2 -= ey_array

            V_array3 = V_array1 + ey_array * y_ext * (-1)
            V_array4 = V_array2 + ey_array * y_ext * (-1)

            V_tot = np.concatenate([V_array1, V_array2, V_array4, V_array3], axis = 1)
            n_tot = np.concatenate([n_array1, n_array2, n_array2, n_array1], axis = 1)

            I_array1 = np.array([
                np.arange(n-1),
                np.arange(1,n),
                np.arange(1,n)+n,
                np.arange(n-1),
                np.arange(1,n)+n,
                np.arange(n-1)+n]).T
            I_array1 = np.concatenate(I_array1)
            I_array_start_cap = np.array([0,n,2*n, 0,2*n,3*n])
            I_array_end_cap =   np.array([0,2*n,n, 0,3*n,2*n])+ (n-1)

            I_tot = np.concatenate([
                I_array1,
                I_array1 + n,
                I_array1 + 2*n,
                I_array1 + 3*n,
                I_array_start_cap,
                I_array_end_cap])
            I_tot = np.mod(I_tot, 4*n)
            C_tot = np.array([0.8,0.8,0.8,1])

            V_opengl = np.zeros(V_tot.shape[1], dtype=vtype)
            V_opengl['a_position'] = V_tot.T
            V_opengl['a_color']    = C_tot
            V_opengl['a_normal']   = n_tot.T
            V.append(V_opengl)
            I.append(I_tot)

        V = np.concatenate(V)
        I = np.concatenate([I[0], I[1] + 4*n])

        return V, I.astype(np.uint32)

    def preview_surface(self) -> None:
        ''' plot surface in 3d to preview shape '''
        window = Window(self)

        while window.draw():
            pass
        window.close()

    def preview_surface_2d(self) -> None:
        ''' create a 2d preview of the surface outline using matplotlib '''
        if not _PYPLOT_AVAILABLE:
            raise RuntimeError('2D plotting requires matplotlib')
        s = np.linspace(self.s_min(), self.s_max(), 1000)
        y_max = self.y_max(s)
        y_min = self.y_min(s)
        left_bnd = self.p2xp(s[None], y_max[None])
        right_bnd = self.p2xp(s[None], y_min[None])
        plt.plot(left_bnd[0], left_bnd[1], 'b')
        plt.plot(right_bnd[0], right_bnd[1], 'b')
        start_bnd = self.p2xp(
            self.s_min(),
            np.linspace(self.y_min(self.s_min()), self.y_max(self.s_min()), 100)[None]
        )
        end_bnd = self.p2xp(
            self.s_max(),
            np.linspace(self.y_min(self.s_max()), self.y_max(self.s_max()), 100)[None]
        )
        plt.plot(start_bnd[0], start_bnd[1], 'b')
        plt.plot(end_bnd[0], end_bnd[1], 'b')

    def pose_eqns_3D(self, vb: ca.SX, wb: ca.SX, rot: Rotation) -> Tuple[ca.SX, ca.SX, ca.SX]:
        '''
        compute pose evolution for unconstrained body
        vb should be a 3x1 variable for velocity in the body frame
        wb should be the same for angular velocity
        returns: p_dot, r_dot
        '''
        xps = self.sym_rep.xps
        xpy = self.sym_rep.xpy
        thp = self.sym_rep.thp

        if rot.ref == Reference.GLOBAL:
            R_rel = self.sym_rep.Rp.T @ rot.R()
        else:
            R_rel = rot.R()

        J3 = ca.vertcat(
            ca.horzcat(ca.norm_2(xps), 0, 0),
            ca.horzcat(-ca.norm_2(xpy) * ca.sin(thp), ca.norm_2(xpy) * ca.cos(thp), 0),
            ca.horzcat(0,0,1)
        ) @ R_rel
        I3 = self.sym_rep.I3
        Q_inv = self.sym_rep.Q_inv

        p_dot = ca.inv(I3) @ J3 @ vb


        s_dot = p_dot[0]
        y_dot = p_dot[1]

        wpn = -s_dot * self.sym_rep.ws - y_dot * self.sym_rep.wy
        wp  = -Q_inv @ self.sym_rep.two @ ca.vertcat(s_dot, y_dot)
        wps =-wp[1]
        wpp = wp[0]
        wp = ca.vertcat(wps, wpp, wpn)

        w_rel = wb - R_rel.T @ wp

        if rot.ref == Reference.PARAMETRIC:
            w_eff = w_rel
        else:
            w_eff = wb

        r_dot = rot.M() @ w_eff

        return p_dot, r_dot, w_rel

    def pose_eqns_2D(self, vb1: ca.SX, vb2: ca.SX, wb3: ca.SX, n:float=0)\
            -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        '''
        compute pose evolution for a tangent contact body
        arguments:
            vb1: longitudinal velocity
            vb2: lateral velocity (+ is left of 1)
            wb3: angular velocity
            ths: relative heading angle
            n: fixed normal offset
        
        returns:
            [s_dot, y_dot, ths_dot, wb1, wb2]
        '''
        one = self.sym_rep.one
        two = self.sym_rep.two
        J   = self.sym_rep.J
        dsdy = ca.inv(one - n * two) @ J @ ca.vertcat(vb1,vb2)
        s_dot = dsdy[0]
        y_dot = dsdy[1]

        # parametric orientation derivative
        ws = self.sym_rep.ws
        wy = self.sym_rep.wy
        ths_dot = wb3 + ws * s_dot + wy * y_dot

        # constrained angular velocity terms
        w2w1 = self.sym_rep.J_inv @ two @ dsdy
        w2 = -w2w1[0]
        w1 = w2w1[1]

        return s_dot, y_dot, ths_dot, w1, w2

    @abstractmethod
    def pose_eqns_2D_planar(self, vb1: ca.SX, vb2: ca.SX, wb3: ca.SX, n:float=0)\
            -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        ''' 2D vehicle pose equations as if the surface were planar '''

@dataclass
class BaseCenterlineSurfaceConfig(BaseSurfaceConfig):
    ''' centerline surface config '''
    s: np.ndarray = field(default = None)
    ''' s keypoints for determining surface shape '''
    n_grid: int = field(default = 10000)
    ''' number of gridpoints to create along the centerline for global -> parametric conversion '''

    x0:  np.ndarray = field(default = None)
    ''' initial position of the surface '''

    y_invariant: bool = field(default = True)

    def __post_init__(self):
        if self.x0 is None:
            self.x0 = np.array([0., 0., 0.])

@dataclass
class BaseCenterlineSurfaceSymRep(BaseSurfaceSymRep):
    ''' centerline surface symbolic representation '''
    xc: Union[ca.SX, ca.MX] = field(default = None)
    ''' centerline '''
    ecs: ca.SX = field(default = None)
    ''' centerline tangent vector '''
    ecy: ca.SX = field(default = None)
    ''' centerline lateral vector '''
    ecn: ca.SX = field(default = None)
    ''' centerlien normal vector '''


class BaseCenterlineSurface(BaseSurface):
    ''' surface that is relative to a centerline '''
    sym_rep: BaseCenterlineSurfaceSymRep
    config: BaseCenterlineSurfaceConfig
    p2xc: Callable[[float], Union[ca.SX, np.ndarray]]
    ''' (s) -> 3D centerline position '''
    p2ecs: Callable[[float], Union[ca.SX, np.ndarray]]
    ''' (s) -> centerline tangent vector '''
    p2ecy: Callable[[float], Union[ca.SX, np.ndarray]]
    ''' (s) -> centerline lateral vector '''
    p2ecn: Callable[[float], Union[ca.SX, np.ndarray]]
    ''' (s) -> centerline normal vector (not the same as epn or xpn)'''

    s_grid: np.ndarray
    ''' s gridpoints used for approxiate global -> parametric conversion'''
    x2s: KDTree
    ''' approximate converter from (xi,xj,xk) to closest (s) coordinate along centerline '''

    def _post_sym_rep(self):
        self._setup_centerline()
        super()._post_sym_rep()

    @abstractmethod
    def _setup_centerline(self):
        ''' set up centerline to define the surface shape '''
        # default implementation of integrating self.sym_rep.esc for an implicit surface
        # override if undesired
        # left abstract to avoid undesired default behavior
        #   child class must specify _setup_centerline implementation
        assert self.config.mx_xp

        s = self.sym_rep.s
        x = ca.SX.sym('x',3)
        es = self.fill_in_param_terms(self.sym_rep.ecs, [s])(s)

        ode = {
            'x': ca.vertcat(s, x),
            'ode': ca.vertcat(1, es)
        }
        s_grid = np.linspace(self.s_min(), self.s_max(), self.config.n_grid)
        config = {'t0':self.s_min(),
                  'tf':self.s_max(),
                  'grid':s_grid,
                  'output_t0':True,
                  'max_step_size':(s_grid[1] - s_grid[0])}
        try:
            # try integrator setup for casadi >= 3.6.0
            xint = ca.integrator('x','idas',ode, self.s_min(), s_grid)
        except NotImplementedError:
            xint = ca.integrator('x','idas',ode, config)

        xc_grid = np.array(xint(x0 = [self.s_min(),*self.config.x0])['xf'])

        xi = ca.interpolant('x','linear',[xc_grid[0]],xc_grid[1])
        xj = ca.interpolant('x','linear',[xc_grid[0]],xc_grid[2])
        xk = ca.interpolant('x','linear',[xc_grid[0]],xc_grid[3])
        s_mx = ca.MX.sym('s')
        y_mx = ca.MX.sym('y')
        n_mx = ca.MX.sym('n')
        p_mx = ca.vertcat(s_mx, y_mx, n_mx)
        xc = ca.vertcat(xi(s_mx),xj(s_mx),xk(s_mx))
        self.sym_rep.xc = xc
        self.sym_rep.p_mx = p_mx

    def _setup_helper_functions(self):
        super()._setup_helper_functions()
        s = self.sym_rep.s
        if not self.config.mx_xp:
            self.p2xc = self.fill_in_param_terms(self.sym_rep.xc, [s])
        else:
            self.p2xc = self.fill_in_param_terms(self.sym_rep.xc, [self.sym_rep.p_mx[0]])

        self.p2ecs = self.fill_in_param_terms(self.sym_rep.ecs, [s])
        self.p2ecy = self.fill_in_param_terms(self.sym_rep.ecy, [s])
        self.p2ecn = self.fill_in_param_terms(self.sym_rep.ecn, [s])

        self.s_grid = np.linspace(self.s_min(), self.s_max(), self.config.n_grid)
        xc_grid = self.p2x_fast(self.s_grid[None], 0, 0).T
        self.x2s = KDTree(xc_grid)

    def g2ps(self, state: BaseBodyState):
        '''
        estimate s coordinate for vehicle by finding closest grid point
        this may not be correct for all possible surfaces and positions
        in particular when in torsion with nonzero normal offset
        '''
        s0 = self.s_grid[self.x2s.query(state.x.to_vec())[1]]
        state.p.s = s0 + float(self.p2ecs(s0) @ (state.x.to_vec() - self.p2xc(s0)))

    def g2px(self, state: BaseBodyState, exact: bool = True):
        # approximate s coordinate
        self.g2ps(state)

        # approximate y coordinate
        delta = (state.x.to_vec() - self.p2xc(state.p.s))
        state.p.y = delta @ self.p2ecy(state.p.s)
        state.p.y = np.clip(state.p.y, self.y_min(state.p.s), self.y_max(state.p.s))

        if exact:
            # optimization refinement for coordinates
            ubp = [self.s_max(), self.y_max(state.p.s), np.inf]
            lbp = [self.s_min(), self.y_min(state.p.s), -np.inf]
            state.p.from_vec(
                self.x2p(state.x.to_vec(), state.p.to_vec(), ubp, lbp)
            )
        else:
            # approximate n coordinate
            delta = (state.x.to_vec() - self.p2xp(state.p.s, state.p.y))
            state.p.n = delta @ self.p2xpn(state.p.s, state.p.y)

    def triangulate_num_s(self) -> int:
        return 1000

    def triangulate_num_y(self) -> int:
        return 30


class BaseLinearCenterlineSurface(BaseCenterlineSurface):
    ''' centerline surface with linear cross section'''

    def _setup_centerline(self):
        assert self.config.mx_xp
        # call parent to integrate centerline if not yet defined
        if self.sym_rep.xc is None:
            super()._setup_centerline()

        s = self.sym_rep.s
        y = self.sym_rep.y
        s_mx = self.sym_rep.p_mx[0]
        y_mx = self.sym_rep.p_mx[1]
        xc = self.sym_rep.xc

        lat_offset = self.fill_in_param_terms(y * self.sym_rep.ecy, [s, y])(s_mx, y_mx)
        xp = xc + lat_offset
        self.sym_rep.xp = xp
