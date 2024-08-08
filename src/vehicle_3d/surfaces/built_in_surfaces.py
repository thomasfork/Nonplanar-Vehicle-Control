''' surfaces that are provided with the libraries, particularly for examples '''
# pylint: disable=line-too-long

import numpy as np
import casadi as ca
from numpy import pi

from vehicle_3d.utils.interp import InterpMethod

# centerline surfaces
from vehicle_3d.surfaces.frenet_surface import FrenetSurfaceConfig
from vehicle_3d.surfaces.tait_bryan_surface import TaitBryanSurfaceConfig
from vehicle_3d.surfaces.arc_profile_centerline_surface import ArcProfileSurfaceConfig
from vehicle_3d.surfaces.poly_profile_centerline_surface import PolyProfileSurfaceConfig
from vehicle_3d.surfaces.banked_spline_surface import BankedSplineSurfaceConfig

# grid surfaces
from vehicle_3d.surfaces.euclidean_surface import EuclideanSurfaceConfig
from vehicle_3d.surfaces.elevation_surface import ElevationSurfaceFunctionConfig, ElevationSurfaceInterpConfig



def l_track():
    ''' an L-shaped track '''
    ds = np.array([0,4,pi,2,pi/2,2,pi,4,pi/2]) * 10
    s = np.cumsum(ds)
    a = np.array([0,0,np.pi, np.pi, np.pi/2, np.pi/2, 3*np.pi/2, 3*np.pi/2, np.pi*2])
    config = FrenetSurfaceConfig(s = s, a = a, y_min = -4, y_max = 4, closed = True)
    return config

def plus():
    ''' a plus-shaped track '''
    ds = np.array([0,2,4,4,2,4,4,4,2,4,4,4,2,4,4,4,2,2]) * 10
    s = np.cumsum(ds)
    a = np.array([0,0,np.pi, np.pi, np.pi/2,np.pi/2, 1.5*np.pi, 1.5*np.pi, np.pi, np.pi, 2*np.pi, 2*np.pi, 1.5*np.pi, 1.5*np.pi, 2.5*np.pi, 2.5*np.pi, 2*np.pi, 2*np.pi])
    config = FrenetSurfaceConfig(s = s, a = a, y_min = -4, y_max = 4, closed = True, a_interp_method=InterpMethod.PCHIP)
    return config

def l_track_smooth():
    ''' a smoothed out l-shaped track'''
    config = BankedSplineSurfaceConfig()
    config.x = np.array([[10,0,0],
                              [20,0,0],
                              [20,10,0],
                              [12,12,0],
                              [10,20,0],
                              [0,20,0],
                              [0,10,0],
                              [0,0,0],
                              [10,0,0]]) * 3
    config.b = -np.array([0.0,0.2,0.2,0.0,0.0,0.0,0.0,0.0,0.0])
    config.closed = True
    config.s = np.arange(config.x.shape[0])
    return config

def lab_track_barc():
    ''' 1/10 scale rc car track '''
    ds = np.array(
        [0,2.251,pi*1.1515,0.901,1.0201675*pi/2,0.15,
        1.2281825*pi,2.25454,1.2206275*pi/2,0.905905])
    s = np.cumsum(ds)
    a = np.array([0,0,pi,pi,pi/2,pi/2,pi*1.5,pi*1.5,pi*2,pi*2]) - pi/2
    config = FrenetSurfaceConfig(s = s, a = a, y_min = -0.55, y_max = 0.55, closed = True)
    config.x0 = [-3.4669, 1.9382,0]
    return config

def lab_track():
    ''' scaled up rc car track '''
    ds = np.array([0,1.55,pi*1.34,2.17,pi*1.34,0.62]) * 10
    s = np.cumsum(ds)
    a = np.array([0,0,pi,pi,2*pi,2*pi])
    config = FrenetSurfaceConfig(s = s, a = a, y_min = -5.95, y_max = 5.95, closed = True)
    return config

def twist():
    ''' a simple twist '''
    ds  = np.array([0,10, 10,     10,     10,      10,      10,      10,      10,       15,     20,       10])
    s = np.cumsum(ds)
    a = np.array([0,  0,      pi/4,   pi/2,   pi/2,   pi/4,   0,      0,      0,      -pi/4,  -pi/2,  -pi/2])
    config = FrenetSurfaceConfig(s = s, a = a, y_min = -4, y_max = 4)
    return config

def hill_climb():
    ''' a simple hill climb '''
    ds  = np.array([0,10, 10,     10,     10,      10,      10,      10,      10,       15,     20,       10])
    s = np.cumsum(ds)
    a = np.array([0,  0,      pi/2,   pi/2,   pi/2,   pi/4,   0,      0,      0,      -pi/2,  -pi/2,  -pi/2])
    b = np.array([0,  0,      0,      0,      0.2,    0.4,    0.5,    0.4,    0.25,    0.1,     0,      0])
    c = np.array([0,  -0.2,   -0.3,   -0.2,   0,      0.2,      0.3,    0.2,    0,      0.2,     0,      0])
    config = TaitBryanSurfaceConfig(s = s, a = a ,b = b, c = c, y_min = -4, y_max = 4)
    return config

def off_camber():
    ''' a single off camber turn '''
    ds = np.array([0, 10,20,20,20,10])
    s = np.cumsum(ds)
    a = np.array([0,0,0,pi/3, pi/3, pi/3])
    b = 0 * a
    c = np.array([0,0,pi/8,pi/8,0,0])
    config = TaitBryanSurfaceConfig(s = s, a = a ,b = b, c = c, y_min = -4, y_max = 4)
    return config

def track_with_loop():
    ''' example used for original nonplanar paper '''
    ds = np.array([0,50, 15,    40,   15,  30,      20,    20,   20,   20,     30,   30            , 10,       30,       10,    10,  10, 20,  15,15,20])
    s  = np.cumsum(ds)
    a  = np.array([0, 0, 0 ,   pi ,  pi,  pi/2,   2.5,     2.5,     2.5 ,  pi/2,     pi/2,  pi  ,  pi,       pi/2,    pi/2,   0,  0,  0, 0, 0,0])
    b  = np.array([0, 0, 0,     0,   0  , 0,      pi/2,    pi,    3*pi/2, 2*pi,    2*pi,   2*pi , 2*pi+1  , 2*pi,  2*pi,  2*pi, 2*pi, 2*pi, 1.7*pi, 2*pi,2*pi] )
    c  = np.array([0, 0, -pi/5, -pi/5, 0,   0,      0,    0,    0,         0,       0,     0,      0.3,        0.1,   -0.3, -0.7,   0,  0  , 0, 0,0])
    config = TaitBryanSurfaceConfig(s = s, a = a ,b = b, c = c, y_min = -4, y_max = 4,
                                    b_interp_method=InterpMethod.LINEAR,
                                    c_interp_method=InterpMethod.LINEAR)
    return config

def track_with_loop_smooth():
    ''' example used for original nonplanar paper with smoother interpolation'''
    ds = np.array([0,50, 15,    40,   15,  30,      20,    20,   20,   20,     30,   30            , 10,       30,       10,    10,  10, 20,  15,15,20])
    s  = np.cumsum(ds)
    a  = np.array([0, 0, 0 ,   pi ,  pi,  pi/2,   2.5,     2.5,     2.5 ,  pi/2,     pi/2,  pi  ,  pi,       pi/2,    pi/2,   0,  0,  0, 0, 0,0])
    b  = np.array([0, 0, 0,     0,   0  , 0,      pi/2,    pi,    3*pi/2, 2*pi,    2*pi,   2*pi , 2*pi+1  , 2*pi,  2*pi,  2*pi, 2*pi, 2*pi, 1.7*pi, 2*pi,2*pi] )
    c  = np.array([0, 0, -pi/5, -pi/5, 0,   0,      0,    0,    0,         0,       0,     0,      0.3,        0.1,   -0.3, -0.7,   0,  0  , 0, 0,0])
    config = TaitBryanSurfaceConfig(s = s, a = a ,b = b, c = c, y_min = -4, y_max = 4)
    return config

def chicane():
    ''' a chicane with on-camber turns '''
    ds = np.array([0, 20,20,   15, 15,   40,   15, 15,20])
    s  = np.cumsum(ds)
    a  = np.array([0, 0, 0,    -pi/6, -pi/3, -pi/3, -pi/6, 0, 0])
    b  = a * 0
    c  = np.array([0, 0, pi/12, pi/6, pi/12, -pi/12, -pi/6, -pi/12, 0])
    config = TaitBryanSurfaceConfig(s = s, a = a ,b = b, c = c, y_min = -6, y_max = 6)
    return config

def tube_turn():
    ''' a quarter pipe turn '''
    config = ArcProfileSurfaceConfig()
    config.s = np.array([0,15,30,50,65,80])*3
    config.a = np.array([0,0,0, np.pi, np.pi,np.pi])
    config.b = np.array([0,0,0,0,0,0])
    config.c = np.array([0,0,0,0,0,0])
    config.k = np.array([.001,.001,.099,0.099,.001,.001])
    config.y_max = 0
    config.y_min = -10
    return config

def tube_turn_closed():
    ''' a quarter pipe turn that loops back '''
    config = ArcProfileSurfaceConfig()
    ds = np.array([0,10,10,20,10,10,5,20,5])
    config.s = np.cumsum(ds) * 3
    config.a = np.array([0,0,0, np.pi, np.pi,np.pi,np.pi,2*np.pi,2*np.pi])
    config.b = np.array([0,0,0,0,0,0,0,0,0])
    config.c = np.array([0,0,0,0,0,0,0,0,0])
    config.k = np.array([.001,.001,.099,0.099,.001,.001,.001,.001,.001])
    config.y_max = 0
    config.y_min = -10
    config.closed = True
    return config

def tube_bend():
    ''' an S-bend with tubular profile '''
    config = ArcProfileSurfaceConfig()
    ds = np.array([0,10,10,20,10,20,10,10])
    config.s = np.cumsum(ds) * 3
    config.a = np.array([0,0,0, np.pi, np.pi,0,0,0])
    config.b = np.array([0,0,0,0,0,0,0,0])
    config.c = np.array([0,0,0,0,0,0,0,0])
    config.k = np.array([.001,.001,.099,0.099,0.099,0.099,.001,.001])
    config.y_max = 10
    config.y_min = -10
    return config

def pump_track():
    ''' a fun track meant for motorcycles '''
    config = PolyProfileSurfaceConfig()
    ds = np.array([0, 10,10, 20, 20, 20, 20, 40, 20,20,20,20,10,10,10,10,10,10,20,40,20,20,60,20,20,30,40,50,44.95,5.877])
    config.s = np.cumsum(ds)
    config.a = np.array([0,0,0,1,1,2.5,2.5,0.5,0.5,1,1,2,2,2,2,2,2,2,3,3,2.5,3,5,5,4.5,4.5,2.5,3,4,4]) * np.pi/2

    config.p0 = np.array([0,0,0,0,0,0,-1,-1,0,0,5,5,4,5,4,5,4,5,5,0,0,0,0,0,0,0,0,0,0,0])
    config.p1 = np.array([0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,.2,-.2,-.2,0,0,.5,.5,-.5,0,0])
    config.p2 = np.array([0,0,0.5,0.5,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-.5,-.5,.5,0,0]) / 6
    config.y_max = 3
    config.y_min = -3
    config.closed = True
    config.a_interp_method = InterpMethod.PCHIP
    return config

def aerial_loop():
    ''' a complicated sequence of aerial features '''
    config = BankedSplineSurfaceConfig()
    config.x = np.array([
        [0,0,0],
        [25,0,0],
        [30,10,-3], # chicane + turn
        [40,10,-3],
        [30,-10,-3],
        [10,-30,0],
        [20,-35,2.5],  # ascent
        [20,0,5],
        [30,20,5],
        [40,10,5],
        [50,10,5],
        [50,20,5],
        [25,25,0],
        [0,20,0],
        [-5,10,0],
        [0,0,0]
    ])
    config.b = np.array([
        0,
        0.2,
        0.2, # chicane + turn
        0.2,
        -.4,
        0,
        -0.2, # ascent
        -0.2,
        -0.2,
        -0.2,
        -0.2,
        0,
        0,
        -0.4,
        0.4,
        0,
    ])
    config.closed = True
    config.s = np.arange(config.x.shape[0])
    return config

def hill():
    ''' a generic hill '''
    config = ElevationSurfaceFunctionConfig()
    return config

def hill_interp():
    ''' a generic hill with different inerpolation setup '''
    config = ElevationSurfaceInterpConfig()
    return config

def slope():
    ''' a short uphill slope '''
    s = ca.SX.sym('s')
    h = 4 * (1 + ca.sin(pi*s/40))
    config = ElevationSurfaceFunctionConfig(sx_h = h, y_min = -4, y_max = 4, s_min = -20, s_max = 20)
    return config

def flat():
    ''' a flat plane '''
    config = EuclideanSurfaceConfig()
    return config
