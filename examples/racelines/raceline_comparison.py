''' example of a few racelines '''

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.raceline.base_raceline import RacelineConfig
from vehicle_3d.raceline.point_raceline import PointRaceline, PointModelConfig
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

point_solver = PointRaceline(
    surf,
    raceline_config.copy(),
    PointModelConfig(),
)

planar_kinematic_solver = PlanarKinematicRaceline(
    surf,
    raceline_config.copy(),
    VehicleModelConfig()
)

kinematic_solver = KinematicRaceline(
    surf,
    raceline_config.copy(),
    VehicleModelConfig()
)

two_track_solver = SlipInputRaceline(
    surf,
    raceline_config.copy(),
    VehicleModelConfig()
)

solvers = [
    point_solver,
    planar_kinematic_solver,
    kinematic_solver,
    two_track_solver,
]

RacelineWindow(
    surf,
    [solver.model for solver in solvers],
    [solver.solve() for solver in solvers]
)
