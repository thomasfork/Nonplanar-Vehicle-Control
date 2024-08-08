'''
arc profile centerline surface
builds on top of Tait-Bryan angle surface
adding an arc normal offset to the centerline
'''
from typing import Tuple
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from vehicle_3d.surfaces.base_surface import BaseCenterlineSurface, \
    BaseCenterlineSurfaceConfig, BaseCenterlineSurfaceSymRep
from vehicle_3d.utils.interp import InterpMethod, interp

@dataclass
class ArcProfileSurfaceConfig(BaseCenterlineSurfaceConfig):
    ''' arc profile centerline surface config'''
    s:   np.ndarray = field(default = np.array([0]))
    a:   np.ndarray = field(default = np.array([0]))
    b:   np.ndarray = field(default = np.array([0]))
    c:   np.ndarray = field(default = np.array([0]))
    k:   np.ndarray = field(default = np.array([0]))

    a_interp_method: InterpMethod = field(default = InterpMethod.LINEAR)
    b_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    c_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    k_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)

    def __post_init__(self):
        super().__post_init__()
        if self.s is None and self.a is None:
            self.s = np.array([0,10,20])
            self.a = np.array([0,0,np.pi/2])
            self.b = np.array([0,0,0])
            self.c = np.array([0,0,0])
            self.k = np.array([0.001,0.1,0.001])

class ArcProfileCenterlineSurface(BaseCenterlineSurface):
    ''' arc profile centerline surface '''
    a: ca.Function = None
    da: ca.Function = None
    dda: ca.Function = None
    b: ca.Function = None
    db: ca.Function = None
    ddb: ca.Function = None
    c: ca.Function = None
    dc: ca.Function = None
    ddc: ca.Function = None
    k: ca.Function = None
    dk: ca.Function = None
    ddk: ca.Function = None

    def __init__(self, config: ArcProfileSurfaceConfig):
        config.flat = np.all(config.b == 0) and np.all(config.c == 0) \
            and np.all(config.k == 0)
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        config.mx_xp = True
        config.orthogonal = False
        super().__init__(config)

    def _setup_interp(self):
        assert isinstance(self.config, ArcProfileSurfaceConfig)
        spline_bc = 'periodic' if self.config.closed else 'not-a-knot'
        self.a = interp(self.config.s, self.config.a, self.config.a_interp_method,
                        spline_bc=spline_bc)
        self.b = interp(self.config.s, self.config.b, self.config.b_interp_method,
                        spline_bc=spline_bc)
        self.c = interp(self.config.s, self.config.c, self.config.c_interp_method,
                        spline_bc=spline_bc)
        self.k = interp(self.config.s, self.config.k, self.config.k_interp_method,
                        spline_bc=spline_bc)

        s = ca.SX.sym('s')
        self.da  = ca.Function('da',  [s], [ca.jacobian(self.a(s), s)])
        self.dda = ca.Function('dda', [s], [ca.jacobian(self.da(s), s)])
        self.db  = ca.Function('db',  [s], [ca.jacobian(self.b(s), s)])
        self.ddb = ca.Function('ddb', [s], [ca.jacobian(self.db(s), s)])
        self.dc  = ca.Function('dc',  [s], [ca.jacobian(self.c(s), s)])
        self.ddc = ca.Function('ddc', [s], [ca.jacobian(self.dc(s), s)])
        self.dk  = ca.Function('dk',  [s], [ca.jacobian(self.k(s), s)])
        self.ddk = ca.Function('ddk', [s], [ca.jacobian(self.dk(s), s)])

    def _compute_sym_rep(self):
        self.sym_rep = BaseCenterlineSurfaceSymRep()
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')

        a = ca.SX.sym('a')
        b = ca.SX.sym('b')
        c = ca.SX.sym('c')
        da = ca.SX.sym('da')
        db = ca.SX.sym('db')
        dc = ca.SX.sym('dc')
        dda = ca.SX.sym('dda')
        ddb = ca.SX.sym('ddb')
        ddc = ca.SX.sym('ddc')
        k = ca.SX.sym('k')
        dk = ca.SX.sym('dk')
        ddk = ca.SX.sym('ddk')

        #rotation matrix for centerline orientation
        Ra = ca.vertcat( ca.horzcat(ca.cos(a),-ca.sin(a),0),
                         ca.horzcat(ca.sin(a), ca.cos(a),0),
                         ca.horzcat(0        , 0        ,1))
        Rb = ca.vertcat( ca.horzcat(ca.cos(b),0,-ca.sin(b)),
                         ca.horzcat(0,        1, 0        ),
                         ca.horzcat(ca.sin(b),0, ca.cos(b)))
        Rc = ca.vertcat( ca.horzcat(1, 0,        0         ),
                         ca.horzcat(0, ca.cos(c),-ca.sin(c)),
                         ca.horzcat(0, ca.sin(c), ca.cos(c)))

        R = Ra @ Rb @ Rc
        es = R[:,0]
        ey = R[:,1]
        en = R[:,2]

        # partial derivatives of centerline basis vectors
        des = ca.jacobian(es, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc)
        dey = ca.jacobian(ey, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc)

        ddey = ca.jacobian(dey, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc) + \
               ca.jacobian(dey, ca.horzcat(da,db,dc)) @ ca.vertcat(dda,ddb,ddc)

        # added normal offset from arc profile term
        p = en * (1 - ca.sqrt(1 - y**2 * k**2))/k

        # partial derivatives of normal offset
        ps = ca.jacobian(p, ca.horzcat(a,b,c,k)) @ ca.vertcat(da,db,dc, dk)
        py = ca.jacobian(p, y)

        pss = ca.jacobian(ps, ca.horzcat(a,b,c,k)) @ ca.vertcat(da,db,dc, dk) + \
              ca.jacobian(ps, ca.horzcat(da,db,dc,dk)) @ ca.vertcat(dda,ddb,ddc, ddk)
        psy = ca.jacobian(ps, y)
        pyy = ca.jacobian(py, y)

        # partial derivatives of parametric surface
        xps = es + y * dey + ps
        xpy = ey + py

        xpss = ddey*y + des + pss
        xpsy = dey  + psy
        xpys = xpsy
        xpyy = ca.SX.zeros(3,1) + pyy


        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(a,b,c,da,db,dc,dda,ddb,ddc,k,dk,ddk)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.a(s),
            self.b(s),
            self.c(s),
            self.da(s),
            self.db(s),
            self.dc(s),
            self.dda(s),
            self.ddb(s),
            self.ddc(s),
            self.k(s),
            self.dk(s),
            self.ddk(s))
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

    def _setup_centerline(self):
        assert self.config.mx_xp
        # call parent to integrate centerline if not yet defined
        if self.sym_rep.xc is None:
            super()._setup_centerline()

        s = self.sym_rep.s
        y = self.sym_rep.y
        k = self.k(s)
        s_mx = self.sym_rep.p_mx[0]
        y_mx = self.sym_rep.p_mx[1]
        xc = self.sym_rep.xc

        lat_offset = self.fill_in_param_terms(y * self.sym_rep.ecy, [s, y])(s_mx, y_mx)
        vert_offset = self.fill_in_param_terms(
            self.sym_rep.ecn * (1 - ca.sqrt(1 - y**2 * k**2))/k,
            [s,y])(s_mx, y_mx)
        xp = xc + lat_offset + vert_offset
        self.sym_rep.xp = xp
