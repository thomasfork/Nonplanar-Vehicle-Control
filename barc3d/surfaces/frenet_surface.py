import numpy as np
import casadi as ca
from matplotlib import pyplot as plt
from dataclasses import dataclass, field

from barc3d.pytypes import  PythonMsg
from barc3d.surfaces.base_surface import BaseLinearCenterlineSurface

@dataclass
class FrenetSurfaceConfig(PythonMsg):
    s:   np.ndarray = field(default = np.array([0]))
    a:   np.ndarray = field(default = np.array([0]))
    y_min: float = field(default = -4)
    y_max: float = field(default = 4)
    
    closed:bool = field(default = False)
    use_pchip: bool = field(default = False) 
    
class FrenetSurface(BaseLinearCenterlineSurface):    
    
    def __init__(self):
        self.config = FrenetSurfaceConfig()
        self.initialized = False

    def _setup_interp(self):
        assert isinstance(self.config, FrenetSurfaceConfig)
        if self.config.use_pchip:
            self.a = self.pchip_interpolant(self.config.s, self.config.a)
        else:
            self.a = ca.interpolant('a', 'linear', [self.config.s], self.config.a)

    def _compute_sym_rep(self):
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        ths = ca.SX.sym('ths')
    
        pose = ca.vertcat(s,y,ths)
    
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
        den = ca.jacobian(en, ca.horzcat(a)) @  ca.vertcat(k)
        
        ddes = ca.jacobian(des, ca.horzcat(a)) @ ca.vertcat(k ) + \
               ca.jacobian(des, ca.horzcat(k )) @ ca.vertcat(dk)
        ddey = ca.jacobian(dey, ca.horzcat(a)) @ ca.vertcat(k ) + \
               ca.jacobian(dey, ca.horzcat(k )) @ ca.vertcat(dk)
        dden = ca.jacobian(den, ca.horzcat(a)) @ ca.vertcat(k ) + \
               ca.jacobian(den, ca.horzcat(k )) @ ca.vertcat(dk)
        
        
        # partial derivatives of parametric surface
        xps = es + y * dey
        xpy = ey
        xpn = ca.cross(xps, xpy)
        xpn = xpn / ca.sqrt(xpn.T @ xpn)
        
        xpss = ddey*y + des 
        xpsy = dey
        xpys = xpsy
        xpyy = ca.SX.zeros(3,1)
    
        # parameterization - dependent terms for symbolic expressions
        param_terms = ca.vertcat(a, k, dk)
        
        
        # same terms but computed from parametric variables
        f_k  = self.a.jacobian()
        f_dk = f_k.jacobian()
        
        s_mx = ca.MX.sym('s')
        y_mx = ca.MX.sym('y')
        ths_mx = ca.MX.sym('ths')
        pose_mx = ca.vertcat(s_mx,y_mx,ths_mx)
        
        param_terms_explicit = ca.vertcat(self.a(s_mx), f_k(s_mx, 0), f_dk(s_mx, 0, 0)[0])
        
        f_param_terms = ca.Function('param_terms',[pose_mx], [param_terms_explicit])
        
        
        sym_rep = dict()
        
        sym_rep['a']   = a
        sym_rep['k']    = k
        sym_rep['dk']   = dk
        
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

