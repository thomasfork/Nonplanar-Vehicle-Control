'''
polynomial profile centerline surface
builds on top of Tait-Bryan angle surface
adding a polynomial normal offset to the centerline
'''
from typing import Tuple
from dataclasses import dataclass, field
import numpy as np
import casadi as ca

from vehicle_3d.surfaces.base_surface import BaseCenterlineSurface, BaseCenterlineSurfaceConfig, \
    BaseCenterlineSurfaceSymRep
from vehicle_3d.utils.interp import InterpMethod, interp

@dataclass
class PolyProfileSurfaceConfig(BaseCenterlineSurfaceConfig):
    ''' polynomial profile centerline surface config'''
    s:   np.ndarray = field(default = np.array([0]))
    a:   np.ndarray = field(default = np.array([0]))
    b:   np.ndarray = field(default = np.array([0]))
    c:   np.ndarray = field(default = np.array([0]))
    p0:  np.ndarray = field(default = np.array([0]))
    p1:  np.ndarray = field(default = np.array([0]))
    p2:  np.ndarray = field(default = np.array([0]))
    p3:  np.ndarray = field(default = np.array([0]))
    p4:  np.ndarray = field(default = np.array([0]))

    a_interp_method: InterpMethod = field(default = InterpMethod.LINEAR)
    b_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    c_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    p0_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    p1_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    p2_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    p3_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)
    p4_interp_method: InterpMethod = field(default = InterpMethod.PCHIP)

class PolyProfileCenterlineSurface(BaseCenterlineSurface):
    ''' polynomial profile centerline surface '''
    a: ca.Function
    da: ca.Function
    dda: ca.Function
    b: ca.Function
    db: ca.Function
    ddb: ca.Function
    c: ca.Function
    dc: ca.Function
    ddc: ca.Function
    p0 : ca.Function
    dp0 : ca.Function
    ddp0 : ca.Function
    p1 : ca.Function
    dp1 : ca.Function
    ddp1 : ca.Function
    p2 : ca.Function
    dp2 : ca.Function
    ddp2 : ca.Function
    p3 : ca.Function
    dp3 : ca.Function
    ddp3 : ca.Function
    p4 : ca.Function
    dp4 : ca.Function
    ddp4 : ca.Function

    def __init__(self, config: PolyProfileSurfaceConfig):
        config.flat = False
        config.s_max = config.s.max()
        config.s_min = config.s.min()
        config.mx_xp = True
        config.orthogonal = False
        super().__init__(config)

    def _setup_interp(self):
        assert isinstance(self.config, PolyProfileSurfaceConfig)

        spline_bc = 'periodic' if self.config.closed else 'not-a-knot'
        self.a = interp(self.config.s, self.config.a, self.config.a_interp_method,
                        spline_bc=spline_bc)
        self.b = interp(self.config.s, self.config.b, self.config.b_interp_method,
                        spline_bc=spline_bc)
        self.c = interp(self.config.s, self.config.c, self.config.c_interp_method,
                        spline_bc=spline_bc)
        self.p0 = interp(self.config.s, self.config.p0, self.config.p0_interp_method,
                         spline_bc=spline_bc)
        self.p1 = interp(self.config.s, self.config.p1, self.config.p1_interp_method,
                         spline_bc=spline_bc)
        self.p2 = interp(self.config.s, self.config.p2, self.config.p2_interp_method,
                         spline_bc=spline_bc)
        self.p3 = interp(self.config.s, self.config.p3, self.config.p3_interp_method,
                         spline_bc=spline_bc)
        self.p4 = interp(self.config.s, self.config.p4, self.config.p4_interp_method,
                         spline_bc=spline_bc)

        s = ca.SX.sym('s')
        self.da  = ca.Function('da',  [s], [ca.jacobian(self.a(s), s)])
        self.dda = ca.Function('dda', [s], [ca.jacobian(self.da(s), s)])
        self.db  = ca.Function('db',  [s], [ca.jacobian(self.b(s), s)])
        self.ddb = ca.Function('ddb', [s], [ca.jacobian(self.db(s), s)])
        self.dc  = ca.Function('dc',  [s], [ca.jacobian(self.c(s), s)])
        self.ddc = ca.Function('ddc', [s], [ca.jacobian(self.dc(s), s)])
        self.dp0  = ca.Function('dp0',  [s], [ca.jacobian(self.p0(s), s)])
        self.ddp0 = ca.Function('ddp0', [s], [ca.jacobian(self.dp0(s), s)])
        self.dp1  = ca.Function('dp1',  [s], [ca.jacobian(self.p1(s), s)])
        self.ddp1 = ca.Function('ddp1', [s], [ca.jacobian(self.dp1(s), s)])
        self.dp2  = ca.Function('dp2',  [s], [ca.jacobian(self.p2(s), s)])
        self.ddp2 = ca.Function('ddp2', [s], [ca.jacobian(self.dp2(s), s)])
        self.dp3  = ca.Function('dp3',  [s], [ca.jacobian(self.p3(s), s)])
        self.ddp3 = ca.Function('ddp3', [s], [ca.jacobian(self.dp3(s), s)])
        self.dp4  = ca.Function('dp4',  [s], [ca.jacobian(self.p4(s), s)])
        self.ddp4 = ca.Function('ddp4', [s], [ca.jacobian(self.dp4(s), s)])

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
        en  =R[:,2]

        # partial derivatives of centerline basis vectors
        des = ca.jacobian(es, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc)
        dey = ca.jacobian(ey, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc)

        ddey = ca.jacobian(dey, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc) + \
               ca.jacobian(dey, ca.horzcat(da,db,dc)) @ ca.vertcat(dda,ddb,ddc)

        # added normal offset from polynomial terms
        p0 = ca.SX.sym('p0')
        p1 = ca.SX.sym('p1')
        p2 = ca.SX.sym('p2')
        p3 = ca.SX.sym('p3')
        p4 = ca.SX.sym('p4')
        p0s = ca.SX.sym('p0s')
        p1s = ca.SX.sym('p1s')
        p2s = ca.SX.sym('p2s')
        p3s = ca.SX.sym('p3s')
        p4s = ca.SX.sym('p4s')
        p0ss = ca.SX.sym('p0ss')
        p1ss = ca.SX.sym('p1ss')
        p2ss = ca.SX.sym('p2ss')
        p3ss = ca.SX.sym('p3ss')
        p4ss = ca.SX.sym('p4ss')

        p = p0*en\
            + y * p1 * en\
            + y**2 * p2 * en \
            + y**3 * p3 * en \
            + y**4 * p4 * en

        # partial derivatives of polynomial offset
        ps = ca.jacobian(p, ca.horzcat(a,b,c)) @ ca.vertcat(da,db,dc) \
            + ca.jacobian(p, ca.horzcat(p0, p1, p2, p3, p4)) @ ca.vertcat(p0s, p1s, p2s, p3s, p4s)
        py = ca.jacobian(p, y)

        pss = ca.jacobian(ps, ca.horzcat(a,b,c)) \
                @ ca.vertcat(da,db,dc) \
            + ca.jacobian(ps, ca.horzcat(da,db,dc)) \
                @ ca.vertcat(dda,ddb,ddc) \
            + ca.jacobian(ps, ca.horzcat(p0,p1,p2,p3,p4)) \
                @ ca.vertcat(p0s,p1s,p2s,p3s,p4s) \
            + ca.jacobian(ps, ca.horzcat(p0s,p1s,p2s,p3s,p4s)) \
                @ ca.vertcat(p0ss,p1ss,p2ss,p3ss,p4ss)
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
        param_terms = ca.vertcat(a,b,c,da,db,dc,dda,ddb,ddc,
                                 p0,p1,p2,p3,p4,
                                 p0s,p1s,p2s,p3s,p4s,
                                 p0ss,p1ss,p2ss,p3ss,p4ss)

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
            self.p0(s),
            self.p1(s),
            self.p2(s),
            self.p3(s),
            self.p4(s),
            self.dp0(s),
            self.dp1(s),
            self.dp2(s),
            self.dp3(s),
            self.dp4(s),
            self.ddp0(s),
            self.ddp1(s),
            self.ddp2(s),
            self.ddp3(s),
            self.ddp4(s),
        )
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
                self.p0(s)
                + self.p1(s)*y
                + self.p2(s)*y**2
                + self.p3(s)*y**3
                + self.p4(s)*y**4),
            [s,y])(s_mx, y_mx)
        xp = xc + lat_offset + vert_offset
        self.sym_rep.xp = xp
