import numpy as np
from numpy import pi as pi

from barc3d.surfaces.base_surface import INTERP_PARTIAL_PCHIP

# centerline surfaces
from barc3d.surfaces.frenet_surface import FrenetSurfaceConfig, FrenetSurface
from barc3d.surfaces.tait_bryan_surface import TaitBryanSurfaceConfig, TaitBryanAngleSurface
from barc3d.surfaces.arc_profile_centerline_surface import ArcProfileCenterlineSurface, ArcProfileSurfaceConfig

def generate_surfaces():
    
    # l_track
    surf = FrenetSurface()
    ds = np.array([0,4,pi,2,pi/2,2,pi,4,pi/2]) * 10
    s = np.cumsum(ds)
    a = np.array([0,0,np.pi, np.pi, np.pi/2, np.pi/2, 3*np.pi/2, 3*np.pi/2, np.pi*2])
    config = FrenetSurfaceConfig(s = s, a = a, y_min = -4, y_max = 4, use_pchip = False, closed = True)
    surf.initialize(config)   
    surf.save_surface('l_track')
    
    
    surf = TaitBryanAngleSurface()
    # itsc
    ds = np.array([0,50, 15,    40,   15,  30,      20,    20,   20,   20,     30,   30            , 10,       30,       10,    10,  10, 20,  15,15,20]) 
    s  = np.cumsum(ds)
    a  = np.array([0, 0, 0 ,   pi ,  pi,  pi/2,   2.5,     2.5,     2.5 ,  pi/2,     pi/2,  pi  ,  pi,       pi/2,    pi/2,   0,  0,  0, 0, 0,0])
    b  = np.array([0, 0, 0,     0,   0  , 0,      pi/2,    pi,    3*pi/2, 2*pi,    2*pi,   2*pi , 2*pi+1  , 2*pi,  2*pi,  2*pi, 2*pi, 2*pi, 1.7*pi, 2*pi,2*pi] )
    c  = np.array([0, 0, -pi/5, -pi/5, 0,   0,      0,    0,    0,         0,       0,     0,      0.3,        0.1,   -0.3, -0.7,   0,  0  , 0, 0,0])
    config = TaitBryanSurfaceConfig(s = s, a = a ,b = b, c = c, y_min = -4, y_max = 4, use_pchip = False)
    surf.initialize(config)   
    surf.save_surface('itsc')
    
    
    # tube turn
    surf = ArcProfileCenterlineSurface()
    config = ArcProfileSurfaceConfig()
    config.s = np.array([0,15,30,50,65,80])*3
    config.a = np.array([0,0,0, np.pi, np.pi,np.pi])
    config.b = np.array([0,0,0,0,0,0])
    config.c = np.array([0,0,0,0,0,0])
    config.k = np.array([.001,.001,.099,0.099,.001,.001])
    config.y_max = 0
    config.y_min = -10
    config.use_pchip = True
    surf.initialize(config)
    surf.save_surface('tube_turn')
    

if __name__ == '__main__':
    generate_surfaces()
    
