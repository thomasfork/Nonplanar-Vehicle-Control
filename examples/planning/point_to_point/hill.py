''' climbing a hill '''
import numpy as np

from vehicle_3d.planning.point_to_point import KinematicPointPlanner, VehicleModelConfig, \
    PointPlannerConfig
from vehicle_3d.visualization.raceline_window import RacelineWindow
from vehicle_3d.surfaces.utils import load_surface

surf = load_surface('hill')
planner = KinematicPointPlanner(
    surf,
    PointPlannerConfig(verbose=True),
    VehicleModelConfig()
)

init_state = planner.model.get_empty_state()
final_state = planner.model.get_empty_state()

init_state.p.s = -30
init_state.ths = -np.pi/2
init_state.p.y = 0
final_state.p.s = 0
final_state.p.y = 0
final_state.ths = np.pi/2

plan = planner.solve(init_state, final_state)

RacelineWindow(surf, planner.model, plan)
