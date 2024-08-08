'''
flat euclidean surface
'''

from dataclasses import dataclass, field

import casadi as ca

from vehicle_3d.pytypes import BaseBodyState
from vehicle_3d.surfaces.base_surface import BaseSurfaceConfig, BaseSurface, \
    BaseSurfaceSymRep


@dataclass
class EuclideanSurfaceConfig(BaseSurfaceConfig):
    '''
    euclidean surface config
    '''
    s_max: float = field(default = 100.)
    s_min: float = field(default =-100.)
    y_max: float = field(default = 100.)
    y_min: float = field(default =-100.)
    n: float = field(default = 0.)

    def __post_init__(self):
        self.closed = False
        self.flat = True
        self.orthogonal = True

class EuclideanSurface(BaseSurface):
    ''' euclidean surface, a flat plane '''
    config: EuclideanSurfaceConfig
    def __init__(self, config: EuclideanSurfaceConfig = None):
        if config is None:
            config = EuclideanSurfaceConfig()
        config.flat = True
        config.closed = False
        config.orthogonal = True
        super().__init__(config)

    def _setup_interp(self):
        pass

    def _compute_sym_rep(self):
        self.sym_rep = BaseSurfaceSymRep()
        s = ca.SX.sym('s')
        y = ca.SX.sym('y')
        n = ca.SX.sym('n')

        xp = ca.vertcat(s, y, self.config.n)
        xps = ca.vertcat(1,0,0)
        xpy = ca.vertcat(0,1,0)
        xpss = ca.vertcat(0,0,0)
        xpsy = ca.vertcat(0,0,0)
        xpys = ca.vertcat(0,0,0)
        xpyy = ca.vertcat(0,0,0)

        param_terms = ca.vertcat([])
        f_param_terms = ca.Function('nuthin',[],[])
        eval_param_terms = ca.vertcat([])

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

    def g2px(self, state: BaseBodyState, exact: bool = True):
        # always exact!
        state.p.from_vec(state.x.to_vec())
        state.p.n -= self.config.n

    def pose_eqns_2D_planar(self, vb1: ca.SX, vb2: ca.SX, wb3: ca.SX, n:float=0):
        return self.pose_eqns_2D(vb1, vb2, wb3, n)
