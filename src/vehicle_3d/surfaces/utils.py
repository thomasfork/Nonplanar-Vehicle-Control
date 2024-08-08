''' utility functions for working with surfaces '''
from typing import Dict
from inspect import isfunction

from vehicle_3d.surfaces import built_in_surfaces

from vehicle_3d.surfaces.base_surface import BaseSurface, BaseSurfaceConfig

from vehicle_3d.surfaces.arc_profile_centerline_surface import ArcProfileCenterlineSurface,\
    ArcProfileSurfaceConfig
from vehicle_3d.surfaces.banked_spline_surface import BankedSplineSurface, \
    BankedSplineSurfaceConfig
from vehicle_3d.surfaces.boundary_spline_surface import BoundarySplineSurface, \
    BoundarySplineSurfaceConfig
from vehicle_3d.surfaces.elevation_surface import ElevationSurface, \
    ElevationSurfaceFunctionConfig, ElevationSurfaceInterpConfig
from vehicle_3d.surfaces.euclidean_surface import EuclideanSurface, \
    EuclideanSurfaceConfig
from vehicle_3d.surfaces.frenet_offset_surface import FrenetOffsetSurface, \
    FrenetOffsetSurfaceConfig, FrenetExpOffsetSurface, FrenetExpOffsetSurfaceConfig
from vehicle_3d.surfaces.frenet_surface import FrenetSurface, \
    FrenetSurfaceConfig
from vehicle_3d.surfaces.poly_profile_centerline_surface import PolyProfileCenterlineSurface, \
    PolyProfileSurfaceConfig
from vehicle_3d.surfaces.spline_surface import SplineSurface, SplineSurfaceConfig
from vehicle_3d.surfaces.tait_bryan_surface import TaitBryanAngleSurface, \
    TaitBryanSurfaceConfig

SURF_CONFIG_MAP : Dict[BaseSurfaceConfig, BaseSurface]= {
    ArcProfileSurfaceConfig: ArcProfileCenterlineSurface,
    BankedSplineSurfaceConfig: BankedSplineSurface,
    BoundarySplineSurfaceConfig: BoundarySplineSurface,
    ElevationSurfaceFunctionConfig: ElevationSurface,
    ElevationSurfaceInterpConfig: ElevationSurface,
    EuclideanSurfaceConfig: EuclideanSurface,
    FrenetOffsetSurfaceConfig: FrenetOffsetSurface,
    FrenetExpOffsetSurfaceConfig: FrenetExpOffsetSurface,
    FrenetSurfaceConfig: FrenetSurface,
    PolyProfileSurfaceConfig: PolyProfileCenterlineSurface,
    SplineSurfaceConfig: SplineSurface,
    TaitBryanSurfaceConfig: TaitBryanAngleSurface
}

def get_available_surfaces():
    ''' return a list of available saved surface '''
    return [label for label in dir(built_in_surfaces) if
            isfunction(getattr(built_in_surfaces, label))]

def load_surface(surface_name: str) -> BaseSurface:
    ''' load a surface, initializing it if found, and returning it'''
    try:
        config = getattr(built_in_surfaces, surface_name)()
    except AttributeError as e:
        raise NotImplementedError(f'Surface {surface_name} is not predefined') from e
    return SURF_CONFIG_MAP[type(config)](config)
