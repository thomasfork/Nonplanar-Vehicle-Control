'''
main script for the paper

Fork, Thomas, H. Eric Tseng, and Francesco Borrelli.
"Vehicle models and optimal control on a nonplanar
surface." arXiv preprint arXiv:2204.09720 (2022).

Available Online: https://arxiv.org/pdf/2204.09720

Disclaimer: This example has been reproduced as faithfully as possible
for exact original code refer to
https://github.com/thomasfork/Nonplanar-Vehicle-Control/tree/2a7992c540ec365f3840d32679159e2a0f37df2e

Racelines are computed with several planar and nonplanar racelines, and then
played in open-loop in an interactive viewer
'''

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.raceline.base_raceline import RacelineConfig
from vehicle_3d.raceline.kinematic_bicycle_raceline import KinematicRaceline,\
    VehicleModelConfig, PlanarKinematicRaceline
from vehicle_3d.raceline.two_track_slip_input_raceline import SlipInputRaceline
from vehicle_3d.visualization.raceline_window import RacelineWindow

surf = load_surface('tube_turn')
raceline_config = RacelineConfig(
    closed = False,
    v0 = 40,
    y0 = -5,
    ths0 = 0,
    v_ws = 40,
)

vehicle_config = VehicleModelConfig(
    m = 2303,
    lf = 1.521,
    lr = 1.499,
    tf = 0.625,
    tr = 0.625,
    h = 0.592,
    w = 1.4,
    bf = 1.8,
    br = 1.8,
    ht = 1.2,
    I1 = 955.9,
    I2 = 5000,
    I3 = 5520.1,
    ua_max = 10,
    ua_min = -10,
    ub_max = 10,
    ub_min = 0,
    uy_max = 0.5,
    uy_min = -0.5,
    print_model_warnings = False)
planar_vehicle_config = vehicle_config.copy()
planar_vehicle_config.build_planar_model = True


planar_kinematic_solver = PlanarKinematicRaceline(
    surf,
    raceline_config.copy(),
    planar_vehicle_config
)

kinematic_solver = KinematicRaceline(
    surf,
    raceline_config.copy(),
    vehicle_config
)

two_track_solver = SlipInputRaceline(
    surf,
    raceline_config.copy(),
    vehicle_config
)

solvers = [
    planar_kinematic_solver,
    kinematic_solver,
    two_track_solver,
]

RacelineWindow(
    surf,
    [solver.model for solver in solvers],
    [solver.solve() for solver in solvers]
)
