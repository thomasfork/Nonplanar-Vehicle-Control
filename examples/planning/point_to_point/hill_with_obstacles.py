''' climbing a hill with some obstacles present '''

from vehicle_3d.planning.point_to_point import KinematicOBCAPointPlanner, VehicleModelConfig, \
    RectangleObstacle, OBCAPointPlannerConfig
from vehicle_3d.visualization.raceline_window import RacelineWindow
from vehicle_3d.surfaces.utils import load_surface

surf = load_surface('hill')
obstacles = [RectangleObstacle() for _ in range(2)]
obstacles[0].x.xi = -15
obstacles[0].x.xj = 3
obstacles[0].w.x1 = 2
obstacles[0].w.x2 = 2
obstacles[0].w.x3 = 10
obstacles[1].x.xi = -30
obstacles[1].x.xj = -3
obstacles[1].w.x1 = 2
obstacles[1].w.x2 = 2
obstacles[1].w.x3 = 10
for obs in obstacles:
    obs.x.xk = float(surf.p2xp(obs.x.xi, obs.x.xj)[2])
    obs.update()

planner = KinematicOBCAPointPlanner(
    surf,
    OBCAPointPlannerConfig(verbose=True),
    VehicleModelConfig(),
    obstacles
)

init_state = planner.model.get_empty_state()
final_state = planner.model.get_empty_state()

init_state.p.s = -40
init_state.ths = 0
init_state.p.y = 0
final_state.p.s = 0
final_state.p.y = 0
final_state.ths = 0

plan = planner.solve(init_state, final_state)

RacelineWindow(surf, planner.model, plan, obstacles=obstacles)
