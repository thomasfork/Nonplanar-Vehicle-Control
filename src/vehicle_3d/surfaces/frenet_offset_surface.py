'''
frenet frame with linear and arc-shaped vertical offsets
'''
from typing import Tuple
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from vehicle_3d.surfaces.base_surface import BaseCenterlineSurface, BaseCenterlineSurfaceConfig,\
    BaseCenterlineSurfaceSymRep
from vehicle_3d.utils.interp import InterpMethod, interp

@dataclass
class FrenetOffsetSurfaceConfig(BaseCenterlineSurfaceConfig):
    ''' frenet offset surface config'''
    s:   np.ndarray = field(default = np.array([0]))
    a:   np.ndarray = field(default = np.array([0])) # track heading
    b:   np.ndarray = field(default = np.array([0])) # cross-sectional slope
    c:   np.ndarray = field(default = np.array([0])) # vertical offset
    k:   np.ndarray = field(default = np.array([0])) # cross-sectional curvature

    a_interp_method: InterpMethod = field(default = InterpMethod.LINEAR)
    b_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    c_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    k_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)

    # optional arrays of 's' to specify more granular interpolation of b,c,k
    # only used if provided
    s_b: np.ndarray = field(default = None)
    s_c: np.ndarray = field(default = None)
    s_k: np.ndarray = field(default = None)

    def __post_init__(self):
        super().__post_init__()
        if self.s is None and self.a is None:
            self.s = np.array([0,10,20])
            self.a = np.array([0,0,np.pi/2])
            self.b = np.array([0,0,0])
            self.c = np.array([0,0,0])
            self.k = np.array([0,0,0])

class FrenetOffsetSurface(BaseCenterlineSurface):
    ''' frenet frame with affine and arc-shaped offsets '''
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

    def __init__(self, config: FrenetOffsetSurfaceConfig):
        config.flat = np.all(config.b == 0) and np.all(config.c == 0) \
            and np.all(config.k == 0)
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        config.mx_xp = True
        config.orthogonal = False
        super().__init__(config)

    def _setup_interp(self):
        assert isinstance(self.config, FrenetOffsetSurfaceConfig)

        s_b = self.config.s_b if self.config.s_b is not None else self.config.s
        s_c = self.config.s_c if self.config.s_c is not None else self.config.s
        s_k = self.config.s_k if self.config.s_k is not None else self.config.s

        spline_bc = 'periodic' if self.config.closed else 'not-a-knot'
        self.a = interp(self.config.s, self.config.a, self.config.a_interp_method,
                        spline_bc=spline_bc)
        self.b = interp(s_b, self.config.b, self.config.b_interp_method,
                        spline_bc=spline_bc)
        self.c = interp(s_c, self.config.c, self.config.c_interp_method,
                        spline_bc=spline_bc)
        self.k = interp(s_k, self.config.k, self.config.k_interp_method,
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
        da = ca.SX.sym('da')
        dda = ca.SX.sym('dda')

        #rotation matrix for centerline orientation
        Ra = ca.vertcat( ca.horzcat(ca.cos(a),-ca.sin(a),0),
                         ca.horzcat(ca.sin(a), ca.cos(a),0),
                         ca.horzcat(0        , 0        ,1))

        R = Ra
        es = R[:,0]
        ey = R[:,1]
        en  =R[:,2]

        # partial derivatives of centerline basis vectors
        des = ca.jacobian(es, a) @  da
        dey = ca.jacobian(ey, a) @  da

        ddey = ca.jacobian(dey, a) @  da + \
               ca.jacobian(dey, da) @ dda

        # added normal offset from linear and arc profile terms
        b = ca.SX.sym('b')
        db = ca.SX.sym('db')
        ddb = ca.SX.sym('ddb')
        c = ca.SX.sym('c')
        dc = ca.SX.sym('dc')
        ddc = ca.SX.sym('ddc')
        k = ca.SX.sym('k')
        dk = ca.SX.sym('dk')
        ddk = ca.SX.sym('ddk')

        p = en * ((1 - ca.sqrt(1 - (y-self.config.y_max)**2 * k**2))/k + b * y + c)

        # partial derivatives of normal offset
        ps = ca.jacobian(p, ca.horzcat(b,c,k)) @ ca.vertcat(db,dc,dk)
        py = ca.jacobian(p, y)

        pss = ca.jacobian(ps, ca.horzcat(b,c,k)) @ ca.vertcat(db,dc, dk) + \
              ca.jacobian(ps, ca.horzcat(db,dc,dk)) @ ca.vertcat(ddb,ddc, ddk)
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
        param_terms = ca.vertcat(a,da,dda,b, db,ddb,c,dc,ddc, k, dk, ddk)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.a(s),
            self.da(s),
            self.dda(s),
            self.b(s),
            self.db(s),
            self.ddb(s),
            self.c(s),
            self.dc(s),
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
            self.sym_rep.ecn * (
                (1 - ca.sqrt(1 - y**2 * k**2))/k + self.b(s) * y + self.c(s)
            ),
            [s,y])(s_mx, y_mx)
        xp = xc + lat_offset + vert_offset
        self.sym_rep.xp = xp

@dataclass
class FrenetExpOffsetSurfaceConfig(BaseCenterlineSurfaceConfig):
    '''
    exponential offset surface config
    n = exp(-b*y - c) * d + e
    '''
    s:   np.ndarray = field(default = np.array([0]))
    a:   np.ndarray = field(default = np.array([0])) # track heading
    b:   np.ndarray = field(default = np.array([0])) # cross-sectional slope
    c:   np.ndarray = field(default = np.array([0])) # vertical offset
    d:   np.ndarray = field(default = np.array([0])) # cross-sectional curvature
    e:   np.ndarray = field(default = np.array([0])) # cross-sectional curvature

    # optional arrays of 's' to specify more granular interpolation of b,c,d,e
    # only used if provided
    s_b: np.ndarray = field(default = None)
    s_c: np.ndarray = field(default = None)
    s_d: np.ndarray = field(default = None)
    s_e: np.ndarray = field(default = None)

    a_interp_method: InterpMethod = field(default = InterpMethod.LINEAR)
    b_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    c_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    d_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    e_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)

class FrenetExpOffsetSurface(BaseCenterlineSurface):
    ''' frenet frame with exponential offset '''
    a: ca.Function = None
    da: ca.Function = None
    dda: ca.Function = None
    b: ca.Function = None
    db: ca.Function = None
    ddb: ca.Function = None
    c: ca.Function = None
    dc: ca.Function = None
    ddc: ca.Function = None
    d: ca.Function = None
    dd: ca.Function = None
    ddd: ca.Function = None
    e: ca.Function = None
    de: ca.Function = None
    dde: ca.Function = None

    def __init__(self, config: FrenetExpOffsetSurfaceConfig):
        config.flat = np.all(config.b == 0) and np.all(config.c == 0) \
            and np.all(config.d == 0) and np.all(config.e == 0)
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        config.mx_xp = True
        config.orthogonal = False
        super().__init__(config)

    def _setup_interp(self):
        assert isinstance(self.config, FrenetExpOffsetSurfaceConfig)

        s_b = self.config.s_b if self.config.s_b is not None else self.config.s
        s_c = self.config.s_c if self.config.s_c is not None else self.config.s
        s_d = self.config.s_d if self.config.s_d is not None else self.config.s
        s_e = self.config.s_e if self.config.s_e is not None else self.config.s

        spline_bc = 'periodic' if self.config.closed else 'not-a-knot'
        self.a = interp(self.config.s, self.config.a, self.config.a_interp_method,
                        spline_bc=spline_bc)
        self.b = interp(s_b, self.config.b, self.config.b_interp_method,
                        spline_bc=spline_bc)
        self.c = interp(s_c, self.config.c, self.config.c_interp_method,
                        spline_bc=spline_bc)
        self.d = interp(s_d, self.config.d, self.config.d_interp_method,
                        spline_bc=spline_bc)
        self.e = interp(s_e, self.config.e, self.config.e_interp_method,
                        spline_bc=spline_bc)

        s = ca.SX.sym('s')
        self.da  = ca.Function('da',  [s], [ca.jacobian(self.a(s), s)])
        self.dda = ca.Function('dda', [s], [ca.jacobian(self.da(s), s)])
        self.db  = ca.Function('db',  [s], [ca.jacobian(self.b(s), s)])
        self.ddb = ca.Function('ddb', [s], [ca.jacobian(self.db(s), s)])
        self.dc  = ca.Function('dc',  [s], [ca.jacobian(self.c(s), s)])
        self.ddc = ca.Function('ddc', [s], [ca.jacobian(self.dc(s), s)])
        self.dd  = ca.Function('dd',  [s], [ca.jacobian(self.d(s), s)])
        self.ddd = ca.Function('ddd', [s], [ca.jacobian(self.dd(s), s)])
        self.de  = ca.Function('de',  [s], [ca.jacobian(self.e(s), s)])
        self.dde = ca.Function('dde', [s], [ca.jacobian(self.de(s), s)])

    def _compute_sym_rep(self):
        self.sym_rep = BaseCenterlineSurfaceSymRep()
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')

        a = ca.SX.sym('a')
        da = ca.SX.sym('da')
        dda = ca.SX.sym('dda')

        #rotation matrix for centerline orientation
        Ra = ca.vertcat( ca.horzcat(ca.cos(a),-ca.sin(a),0),
                         ca.horzcat(ca.sin(a), ca.cos(a),0),
                         ca.horzcat(0        , 0        ,1))

        R = Ra
        es = R[:,0]
        ey = R[:,1]
        en  =R[:,2]

        # partial derivatives of centerline basis vectors
        des = ca.jacobian(es, a) @  da
        dey = ca.jacobian(ey, a) @  da

        ddey = ca.jacobian(dey, a) @  da + \
               ca.jacobian(dey, da) @ dda

        # added normal offset from linear and arc profile terms
        b = ca.SX.sym('b')
        db = ca.SX.sym('db')
        ddb = ca.SX.sym('ddb')
        c = ca.SX.sym('c')
        dc = ca.SX.sym('dc')
        ddc = ca.SX.sym('ddc')
        d = ca.SX.sym('d')
        dd = ca.SX.sym('dd')
        ddd = ca.SX.sym('ddd')
        e = ca.SX.sym('e')
        de = ca.SX.sym('de')
        dde = ca.SX.sym('dde')

        p = en * ca.exp(-b * y - c) * d + e

        # partial derivatives of normal offset
        ps = ca.jacobian(p, ca.horzcat(b,c,d,e)) @ ca.vertcat(db,dc,dd,de)
        py = ca.jacobian(p, y)

        pss = ca.jacobian(ps, ca.horzcat(b,c,d,e)) @ ca.vertcat(db,dc,dd,de) + \
              ca.jacobian(ps, ca.horzcat(db,dc,dd,de)) @ ca.vertcat(ddb,ddc,ddd,dde)
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
        param_terms = ca.vertcat(a,da,dda,b, db,ddb,c,dc,ddc,d,dd,ddd,e,de,dde)

        # same terms but computed from parametric variables
        param_terms_explicit = ca.vertcat(
            self.a(s),
            self.da(s),
            self.dda(s),
            self.b(s),
            self.db(s),
            self.ddb(s),
            self.c(s),
            self.dc(s),
            self.ddc(s),
            self.d(s),
            self.dd(s),
            self.ddd(s),
            self.e(s),
            self.de(s),
            self.dde(s))
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
        s_mx = self.sym_rep.p_mx[0]
        y_mx = self.sym_rep.p_mx[1]
        xc = self.sym_rep.xc

        lat_offset = self.fill_in_param_terms(y * self.sym_rep.ecy, [s, y])(s_mx, y_mx)
        vert_offset = self.fill_in_param_terms(
            self.sym_rep.ecn * (
                ca.exp(-1*self.b(s) * y - self.c(s)) * self.d(s) + self.e(s)
            ),
            [s,y])(s_mx, y_mx)
        xp = xc + lat_offset + vert_offset
        self.sym_rep.xp = xp
