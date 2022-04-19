'''
Script for demonstrating raceline computation and evaluation
Primarily meant for AVEC 2022 publication
'''
import numpy as np

from barc3d.surfaces import load_surface
from barc3d.dynamics.dynamics_3d import VehicleConfig
from barc3d.raceline.base_raceline import RacelineConfig

from barc3d.raceline.planar_kinematic_bicycle_raceline import PlanarKinematicBicycleRaceline

from barc3d.raceline.kinematic_bicycle_raceline import KinematicBicycleRaceline

from barc3d.raceline.dynamic_bicycle_raceline import DynamicBicycleRaceline

from barc3d.raceline.dynamic_two_track_slip_input_raceline import DynamicTwoTrackSlipInputRaceline
from barc3d.surfaces.base_surface import BaseSurface

def compare(surface: BaseSurface,
            config: RacelineConfig,
            vehicle_config: VehicleConfig,
            pyplot = False,
            details = False,
            include_planar = False):
    '''
    Compares various models on a surface with provided raceline and vehicle configuration
    '''

    surf = load_surface(surface)
    config.closed = surf.config.closed
    results = []

    if include_planar:
        solver = PlanarKinematicBicycleRaceline(surf, config, vehicle_config)
        planar_kinematic_bicycle_results  = solver.solve()
        results.append(planar_kinematic_bicycle_results)

    solver = KinematicBicycleRaceline(surf, config, vehicle_config)
    kinematic_bicycle_results = solver.solve()
    results.append(kinematic_bicycle_results)

    solver = DynamicBicycleRaceline(surf, config, vehicle_config)
    dynamic_bicycle_results = solver.solve()
    results.append(dynamic_bicycle_results)

    config = config.copy()
    config.R = np.eye(5) * 0.001
    config.dR = np.eye(5) * 0.001
    solver = DynamicTwoTrackSlipInputRaceline(surf, config, vehicle_config)
    dynamic_two_track_results = solver.solve()
    results.append(dynamic_two_track_results)

    if not pyplot:
        solver.plot_raceline_3D(results)
    else:
        solver.plot_raceline_pyplot(results, block = not details)

    if details:
        solver.plot_raceline_two_track_details(dynamic_two_track_results)

def main():
    '''
    Examples of model raceline comparison
    '''
    raceline_config = RacelineConfig(verbose = True)
    vehicle_config = VehicleConfig()

    # the planar model comparison
    #compare('l_track', raceline_config, vehicle_config, pyplot = True, details = True)

    # the nonplanar model comparison
    raceline_config.v0 = 40
    raceline_config.y0 = -5
    raceline_config.ths0 = 0
    raceline_config.v_ws = 40
    compare('tube_turn', raceline_config, vehicle_config, pyplot = False, details = False, include_planar = True)
    
if __name__ == '__main__':
    main()
