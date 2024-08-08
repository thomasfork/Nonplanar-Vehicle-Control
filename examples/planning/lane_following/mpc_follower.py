'''
example of a simple MPC setup for lane following
'''
from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.planning.lane_follower import KinematicLaneFollower, LaneFollowerConfig, \
    VehicleModelConfig
from vehicle_3d.visualization.window import Window
surf = load_surface('off_camber')

planner = KinematicLaneFollower(
    surf,
    LaneFollowerConfig(compile=True),
    VehicleModelConfig()
)
planner.v_ref = 8

state = planner.model.get_empty_state()

planner.step(state)

window = Window(surf)
for label, item in planner.model.generate_visual_assets(window.ubo).items():
    window.add_object(label, item)
window.camera_follow=True

plan = planner.get_plan().triangulate_trajectory(
    window.ubo,
    planner.model,
    surf,
    n = 200
)
window.add_object('Current Plan', plan)

planner.model.update_visual_assets(state)
while window.step(state):
    planner.model.zu2state(
        state,
        planner.get_plan().z_interp(state.t + .1),
        planner.get_plan().u_interp(state.t + .1))
    state.t += 0.1

    # wrap s coordinate even for aperiodic surfaces so loop continues
    if state.p.s > surf.s_max():
        state.p.s = surf.s_min()

    planner.step(state)
    print(f'{1000 * planner.current_plan.solve_time:0.2f}ms '
            + ('Feasible' if planner.get_plan().feasible else 'Infeasible'))

    planner.get_plan().update_triangulated_trajectory(plan, planner.model, surf, n = 200)
    planner.model.update_visual_assets(state)
