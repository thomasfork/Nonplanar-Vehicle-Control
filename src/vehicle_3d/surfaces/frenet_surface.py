'''
frenet frame
'''
from typing import Tuple
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from vehicle_3d.utils.interp import InterpMethod, interp
from vehicle_3d.surfaces.base_surface import BaseCenterlineSurfaceConfig, \
    BaseCenterlineSurfaceSymRep, BaseLinearCenterlineSurface


@dataclass
class FrenetSurfaceConfig(BaseCenterlineSurfaceConfig):
    '''
    frenet frame surface defiend in terms of arc length and heading angle
    '''
    s:   np.ndarray = field(default = None)
    a:   np.ndarray = field(default = None)
    a_interp_method: InterpMethod = field(default = InterpMethod.LINEAR)

    def __post_init__(self):
        super().__post_init__()
        if self.s is None and self.a is None:
            self.s = np.array([0,10,20])
            self.a = np.array([0,0,np.pi/2])


class FrenetSurface(BaseLinearCenterlineSurface):
    ''' canonical arc-length and heading frenet frame '''
    a: ca.Function
    da: ca.Function
    dda: ca.Function

    config: FrenetSurfaceConfig

    def __init__(self, config: FrenetSurfaceConfig):
        config.flat = True
        config.mx_xp = True
        config.orthogonal = True
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        super().__init__(config)

    def _setup_interp(self):
        spline_bc = 'periodic' if self.config.closed else 'not-a-knot'
        self.a = interp(self.config.s, self.config.a, self.config.a_interp_method,
                        spline_bc=spline_bc)
        s = ca.SX.sym('s')
        self.da  = ca.Function('da',  [s], [ca.jacobian(self.a(s), s)])
        self.dda  = ca.Function('dda',  [s], [ca.jacobian(self.da(s), s)])

    def _compute_sym_rep(self):
        self.sym_rep = BaseCenterlineSurfaceSymRep()
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')

        a = ca.SX.sym('a')
        k  = ca.SX.sym('k')
        dk = ca.SX.sym('dk')

        #rotation matrix for centerline orientation
        R  = ca.vertcat( ca.horzcat(ca.cos(a),-ca.sin(a),0),
                         ca.horzcat(ca.sin(a), ca.cos(a),0),
                         ca.horzcat(0         , 0         ,1))
        es = R[:,0]
        ey = R[:,1]
        en  =R[:,2]

        # partial derivatives of centerline basis vectors
        des = ca.jacobian(es, ca.horzcat(a)) @  ca.vertcat(k)
        dey = ca.jacobian(ey, ca.horzcat(a)) @  ca.vertcat(k)

        ddey = ca.jacobian(dey, ca.horzcat(a)) @ ca.vertcat(k ) + \
               ca.jacobian(dey, ca.horzcat(k )) @ ca.vertcat(dk)

        # partial derivatives of parametric surface
        xps = es + y * dey
        xpy = ey

        xpss = ddey*y + des
        xpsy = dey
        xpys = xpsy
        xpyy = ca.SX.zeros(3,1)


        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(a, k, dk)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.a(s),
            self.da(s),
            self.dda(s))
        f_param_terms = ca.Function('param_terms',[s, y, n], [param_terms_explicit])

        eval_param_terms = f_param_terms(s, 0, 0)

        self.sym_rep.s = s
        self.sym_rep.y = y
        self.sym_rep.n = n
        self.sym_rep.xps = xps
        self.sym_rep.xpy = xpy
        self.sym_rep.xpss = xpss
        self.sym_rep.xpsy = xpsy
        self.sym_rep.xpys = xpys
        self.sym_rep.xpyy = xpyy

        self.sym_rep.ecs = es
        self.sym_rep.ecy = ey
        self.sym_rep.ecn = en

        self.sym_rep.param_terms = param_terms
        self.sym_rep.f_param_terms = f_param_terms
        self.sym_rep.eval_param_terms = eval_param_terms

    def pose_eqns_2D_planar(self, vb1: ca.SX, vb2: ca.SX, wb3: ca.SX, n:float=0)\
            -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX, ca.SX]:
        y = self.sym_rep.y
        ths = self.sym_rep.ths
        kappa = self.da(self.sym_rep.s)

        s_dot = (vb1 * ca.cos(ths) - vb2 * ca.sin(ths)) / (1- kappa * y)
        y_dot =  vb1 * ca.sin(ths) + vb2 * ca.cos(ths)
        ths_dot = wb3 - kappa * s_dot
        w1 = 0
        w2 = 0
        return s_dot, y_dot, ths_dot, w1, w2

MIN_R = 1 # for automatically trying to shrink radius for xy setup
def from_xy(xy: np.ndarray, R: float = 7, xk=0) -> FrenetSurface:
    '''
    set up a frenet frame track from an array of xy points
    by adding a fillet at every corner
    xy should be of shape Nx2
    '''
    eps = 1e-3
    tht0 = np.arctan2(xy[1,1] - xy[0,1], xy[1,0] - xy[0,0])

    ds = []
    k  = []
    tht = [tht0]

    x_prev = xy[0]

    # pathological case of one straight segment
    if xy.shape[0] == 2:
        s = np.array([0, np.linalg.norm(xy[1] - xy[0])])
        k = np.array([0,0])
        tht = np.array([tht0, tht0])

        config = FrenetSurfaceConfig(
            s = s,
            a = tht,
            x0 = np.array([xy[0,0], xy[0,1], xk])
        )
        return FrenetSurface(config)


    for idx in range(1, xy.shape[0]-1):
        x_cur = xy[idx]
        x_next = xy[idx + 1]

        x1 = x_prev - x_cur
        x2 = x_next - x_cur

        es = np.array([np.cos(tht[-1]), np.sin(tht[-1])])
        # es is not always parallel to x1,
        # use -es for e1 to avoid accumulating this error
        e1 = -es
        e2 = x2 / np.linalg.norm(x2)

        # get a transverse vector from e2 for finding center of arc
        e2t = np.array([[0,-1],[1,0]]) @ e2

        th_next = np.arctan2(x2[1], x2[0])
        # opening angle of corner
        dth = th_next - tht[-1]

        while dth > np.pi:
            dth -= 2*np.pi
        while dth <-np.pi:
            dth += 2*np.pi

        # only add an arc if angle is sufficiently large
        if abs(dth) > eps:
            # length of corner edge to fillet
            d = abs(R / np.tan((np.pi-dth)/2))

            # distance from current centerline point in direction
            # es to intersect next line, not necessarily at x_cur
            # ideally this equals d1, but not necessarily
            d_prev = np.dot(x1, e2t) / np.sin(dth)

            # resulting line intersection
            # this is the first point to add - the result of moving straight forwards to the
            # start of the fillet
            x_int = x_prev + d_prev * es

            # added length to next edge
            d2_added = np.dot(e2, x_cur - x_int)

            if d_prev - d < 0:
                print('WARNING - negative length frenet frame segment, \
                    the radius is probably too large')
                if R < MIN_R:
                    return

                print(f'shrinking radius to {(R*0.8):0.3f}')
                return from_xy(xy, R = R*0.8, xk=xk)

            # straight segment before arc
            if np.linalg.norm(d_prev - d) > eps:
                ds.append(d_prev - d)
                k.append(0)
                tht.append(tht[-1])

            # arc segment
            ds.append(R * np.abs(dth))
            k.append(1/R * np.sign(dth))
            tht.append(tht[-1] + ds[-1] * k[-1])

            # compute the next xy point's resulting location for the next iteration
            th_next = np.arctan2(x2[1], x2[0])

            e_center = (e1 + e2) / np.linalg.norm(e1 + e2)
            d_center = R / np.cos(dth/2)

            x_center = x_int + d_center * e_center

            r1_center = x_prev - x_center
            r2_center = x_int + (d + d2_added) * e2 - x_center

            arc_th_1 = np.arctan2(r1_center[1], r1_center[0])
            arc_th_2 = np.arctan2(r2_center[1], r2_center[0])

            if np.sign(dth) > 0:
                while arc_th_1 > arc_th_2:
                    arc_th_2 += 2*np.pi
            else:
                while arc_th_1 < arc_th_2:
                    arc_th_2 -= 2*np.pi

            x_prev = np.array([np.cos(arc_th_2), np.sin(arc_th_2)]).T * R + x_center



        # final segment at end
        if idx == xy.shape[0] - 2:

            es = np.array([np.cos(tht[-1]), np.sin(tht[-1])])
            d = es.T @ (xy[-1] - x_prev)
            if d > eps:
                ds.append(d)
                k.append(0)
                tht.append(tht[-1])

    s = np.array([0, *np.cumsum(ds)])
    k = np.array([*k, 0])
    tht = np.array(tht)

    config = FrenetSurfaceConfig(
        s = s,
        a = tht,
        x0 = np.array([xy[0,0], xy[0,1], xk])
    )
    return FrenetSurface(config)
