'''
example of a racetrack import
track data is from https://github.com/TUMFTM/racetrack-database/blob/master/tracks/Austin.csv
'''
import os

import numpy as np

from vehicle_3d.surfaces.spline_surface import SplineSurface, SplineSurfaceConfig

path = os.path.dirname(__file__)
filename = os.path.join(path, 'Austin.csv')
data = np.genfromtxt(filename, delimiter=',')

x = np.hstack([data[:,:2], np.zeros((data.shape[0],1))])
y_min =-data[:,2]
y_max = data[:,3]

surf_config = SplineSurfaceConfig(
    closed = True,
    x = x,
    y_max_interp=y_max,
    y_min_interp=y_min,
)
surf = SplineSurface(surf_config)
surf.preview_surface()
