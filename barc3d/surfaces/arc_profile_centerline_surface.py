import numpy as np
import pdb
import casadi as ca
import scipy.interpolate
from matplotlib import pyplot as plt
from dataclasses import dataclass, field

from barc3d.pytypes import  PythonMsg
from barc3d.surfaces.base_surface import BaseCenterlineSurface, INTERP_PARTIAL_PCHIP
from barc3d.pytypes import VehicleState

@dataclass
class ArcProfileSurfaceConfig(PythonMsg):
    s:   np.ndarray = field(default = np.array([0]))
    a:   np.ndarray = field(default = np.array([0]))
    b:   np.ndarray = field(default = np.array([0]))
    c:   np.ndarray = field(default = np.array([0]))
    k:   np.ndarray = field(default = np.array([0]))
    y_min: float = field(default = -4)
    y_max: float = field(default = 4)
    
    closed:bool = field(default = False)
    use_pchip: bool = field(default = False) 
    
    
class ArcProfileCenterlineSurface(BaseCenterlineSurface):    
    '''
    Uses tait bryan angles to define a centerline, on top of which a arc profile is added (curvature k)
    
    '''
    def __init__(self):
        self.config = ArcProfileSurfaceConfig()
        self.initialized = False
    
    def _setup_interp(self):
        assert isinstance(self.config, ArcProfileSurfaceConfig)
        
        if not self.config.use_pchip:
            self.tha = ca.interpolant('tha', 'linear', [self.config.s], self.config.a)
            self.thb = ca.interpolant('thb', 'linear', [self.config.s], self.config.b)  
            self.thc = ca.interpolant('thc', 'linear', [self.config.s], self.config.c)
            self.k   = ca.interpolant('k',   'linear', [self.config.s], self.config.k)
        elif self.config.use_pchip == INTERP_PARTIAL_PCHIP:
            self.tha = ca.interpolant('tha', 'linear', [self.config.s], self.config.a)
            self.thb = ca.interpolant('thb', 'linear', [self.config.s], self.config.b)  
            self.thc = ca.interpolant('thc', 'linear', [self.config.s], self.config.c)
            self.k   = self.pchip_interpolant(self.config.s, self.config.k)
        else:
            self.tha = self.pchip_interpolant(self.config.s, self.config.a)
            self.thb = self.pchip_interpolant(self.config.s, self.config.b)
            self.thc = self.pchip_interpolant(self.config.s, self.config.c)
            self.k   = self.pchip_interpolant(self.config.s, self.config.k)
        
        return
        
    def _compute_sym_rep(self):
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        ths = ca.SX.sym('ths')
    
        pose = ca.vertcat(s,y,ths)
    
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
        den = ca.jacobian(en, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc)
        
        ddes = ca.jacobian(des, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc) + \
               ca.jacobian(des, ca.horzcat(da,db,dc)) @ ca.vertcat(dda,ddb,ddc)
        ddey = ca.jacobian(dey, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc) + \
               ca.jacobian(dey, ca.horzcat(da,db,dc)) @ ca.vertcat(dda,ddb,ddc)
        dden = ca.jacobian(den, ca.horzcat(a,b,c)) @  ca.vertcat(da,db,dc) + \
               ca.jacobian(den, ca.horzcat(da,db,dc)) @ ca.vertcat(dda,ddb,ddc)
        
        
        # added normal offset from arc profile term
        k = ca.SX.sym('k')
        dk = ca.SX.sym('dk')
        ddk = ca.SX.sym('ddk')
        
        p = en * (1 - ca.sqrt(1 - y**2 * k**2))/k
        
        # partial derivatives of normal offset
        ps = ca.jacobian(p, ca.horzcat(a,b,c,k)) @ ca.vertcat(da,db,dc, dk)
        py = ca.jacobian(p, y)
        
        pss = ca.jacobian(ps, ca.horzcat(a,b,c,k)) @ ca.vertcat(da,db,dc, dk) + ca.jacobian(ps, ca.horzcat(da,db,dc,dk)) @ ca.vertcat(dda,ddb,ddc, ddk) 
        psy = ca.jacobian(ps, y)
        pyy = ca.jacobian(py, y)
        
        
        # partial derivatives of parametric surface
        xps = es + y * dey + ps
        xpy = ey + py
        xpn = ca.cross(xps, xpy)
        
        xpss = ddey*y + des + pss
        xpsy = dey  + psy
        xpys = xpsy
        xpyy = ca.SX.zeros(3,1) + pyy
        
        
        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(a,b,c,da,db,dc,dda,ddb,ddc, k, dk, ddk)
        
        # same terms but computed from parametric variables
        
        f_dtha = self.tha.jacobian()
        f_dthb = self.thb.jacobian()
        f_dthc = self.thc.jacobian()
        f_ddtha = f_dtha.jacobian()
        f_ddthb = f_dthb.jacobian()
        f_ddthc = f_dthc.jacobian()
        
        f_dk = self.k.jacobian()
        f_ddk = f_dk.jacobian()
        
        
        s_mx = ca.MX.sym('s')
        y_mx = ca.MX.sym('y')
        ths_mx = ca.MX.sym('ths')
    
        pose_mx = ca.vertcat(s_mx,y_mx,ths_mx)
        
        
        param_terms_explicit = ca.vertcat(self.tha(s_mx), self.thb(s_mx), self.thc(s_mx), \
                                          f_dtha(s_mx,0), f_dthb(s_mx,0), f_dthc(s_mx,0), \
                                          f_ddtha(s_mx,0,0)[0], f_ddthb(s_mx,0,0)[0], f_ddthc(s_mx,0,0)[0],\
                                          self.k(s_mx), \
                                          f_dk(s_mx,0), \
                                          f_ddk(s_mx,0,0)[0])
        f_param_terms = ca.Function('param_terms',[pose_mx], [param_terms_explicit])
        
        sym_rep = dict()
        
        sym_rep['a']   = a
        sym_rep['b']   = b
        sym_rep['c']   = c
        sym_rep['da']   = da
        sym_rep['db']   = db
        sym_rep['dc']   = dc
        sym_rep['dda']   = dda
        sym_rep['ddb']   = ddb
        sym_rep['ddc']   = ddc
        
        sym_rep['k']   = k
        sym_rep['dk']   = dk
        sym_rep['ddk']   = ddk
        
        sym_rep['s']   = s
        sym_rep['y']   = y
        sym_rep['ths'] = ths
        sym_rep['pose'] = pose
        
        sym_rep['s_mx']   = s_mx
        sym_rep['y_mx']   = y_mx
        sym_rep['ths_mx'] = ths_mx
        sym_rep['pose_mx'] = pose_mx
        
        sym_rep['es'] = es
        sym_rep['ey'] = ey
        sym_rep['en'] = en
        
        sym_rep['xps'] = xps
        sym_rep['xpy'] = xpy
        
        sym_rep['xpyy'] = xpyy
        sym_rep['xpys'] = xpys
        sym_rep['xpsy'] = xpsy
        sym_rep['xpss'] = xpss
        
        sym_rep['param_terms'] = param_terms
        sym_rep['f_param_terms'] = f_param_terms
        
        self.sym_rep = sym_rep

    def _post_centerline(self):
        ''' 
        obtain a function from parametric position to global position 
        for a linear centerline surface
        '''
        xc = self.sym_rep['xc']
        ey = self.sym_rep['ey']
        en = self.sym_rep['en']
        
        pose = self.sym_rep['pose']
        pose_mx = self.sym_rep['pose_mx']
        param_terms = self.sym_rep['param_terms']
        f_param_terms = self.sym_rep['f_param_terms']
        
        y = self.sym_rep['y']
        k = self.sym_rep['k']
        f_dy = ca.Function('ey', [pose, param_terms], [ey*y + en * (1-ca.sqrt(1-y**2*k**2))/k])  # general function for lateral offset
        dy = f_dy.call([pose_mx, f_param_terms(pose_mx)])[0]   # lateral offset for specific surface parameterization
        
        # add lateral offset to centerline and turn into a function
        xp = xc + dy
        f_xp = ca.Function('xp',[pose_mx],[xp])
        
        self.sym_rep['xp'] = xp
        self.sym_rep['f_xp'] = f_xp
        self.pose2xp = lambda pose: np.array(f_xp(pose))

