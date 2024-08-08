''' demo of speed planner '''
import imgui
import numpy as np

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.visualization.window import Window
from vehicle_3d.visualization.utils import plot_multiline, IMGUI_WHITE, IMGUI_RED, \
    IMGUI_GREEN, IMGUI_BLUE, IMGUI_YELLOW
from vehicle_3d.planning.speed_planner import SpeedPlanner, SpeedPlannerConfig
from vehicle_3d.models.tangent_vehicle_model import DynamicTwoTrackSlipInputVehicleModel, \
    VehicleModelConfig


TRACE_LEN = 50

sim_surf = load_surface('hill')

sim_config = VehicleModelConfig()
sim = DynamicTwoTrackSlipInputVehicleModel(sim_config, sim_surf)
sim_state = sim.get_empty_state()

plan_config = SpeedPlannerConfig(yaw_input=True, N = 50)
planner = SpeedPlanner(sim_surf, sim_config, plan_config)
delta = np.array([1.])
kappa = np.array([0.])
planner_s = sim_state.p.s + np.arange(planner.config.N) * delta * np.cos(sim_state.ths)
planner_y = sim_state.p.y + np.arange(planner.config.N) * delta * np.sin(sim_state.ths)
planner_ths = sim_state.ths * np.ones(planner.config.N)
planner.solve(sim_state.vb.mag(), planner_s, planner_y, planner_ths)

window = Window(sim_surf)
for label, item in sim.generate_visual_assets(window.ubo).items():
    window.add_object(label, item)
vis_plan = planner.current_plan.triangulate_plan(window.ubo, sim_surf)
vis_plan_points = planner.current_plan.triangulate_plan_points(window.ubo, sim_surf)
window.add_object('Speed Plan', vis_plan)
window.add_object('Speed Plan Points', vis_plan_points)

solve_times = np.zeros(TRACE_LEN)
prep_times  = np.zeros(TRACE_LEN)
unpack_times = np.zeros(TRACE_LEN)
total_times = np.zeros(TRACE_LEN)

def _window_extras():
    changed_1, sim_state.p.s = imgui.slider_float('s', sim_state.p.s,
        min_value = sim_surf.s_min(), max_value = sim_surf.s_max())
    changed_2, sim_state.p.y = imgui.slider_float('y', sim_state.p.y,
        min_value = sim_surf.y_min(), max_value = sim_surf.y_max())
    changed_3, sim_state.ths = imgui.slider_float('ths', sim_state.ths,
        min_value = -np.pi, max_value = np.pi)
    changed_4, sim_state.vb.v1 = imgui.slider_float('v0', sim_state.vb.v1,
        min_value = 0, max_value = planner.config.v_max)
    changed_5, delta[:] = imgui.slider_float('delta', delta[0],
        min_value = .1, max_value = 10)
    changed_6, kappa[:] = imgui.slider_float('kappa', kappa[0],
        min_value = -.1, max_value = .1)

    imgui.radio_button('Feasible', planner.current_plan.feasible)
    imgui.input_float('Max friction usage', planner.current_plan.max_mu)
    imgui.input_float('Min normal force', planner.current_plan.min_N)
    imgui.input_float('Max speed', planner.current_plan.v_max)
    imgui.input_float('Min speed', planner.current_plan.v_min)

    speed_profile_tau = np.linspace(0, 1, planner.config.N)
    speed_profile = planner.current_plan.v_interp(speed_profile_tau)

    plot_multiline(
        [speed_profile_tau],
        [speed_profile],
        [None],
        [None],
        [IMGUI_WHITE],
        ['Speed Planner'],
        'Speed Profile (m/s)',
        y_min = 0,
    )
    plot_multiline(
        [np.arange(TRACE_LEN)]*4,
        [prep_times,solve_times,unpack_times,total_times],
        [None],
        [None],
        [IMGUI_YELLOW, IMGUI_RED, IMGUI_GREEN, IMGUI_BLUE],
        ['Prep','Solver','Unpack','Total'],
        'Solve Timing (us)',
    )
    plot_multiline(
        [speed_profile_tau]*4,
        [
            planner.current_plan.p[2::6],
            planner.current_plan.p[3::6],
            planner.current_plan.p[4::6],
            planner.current_plan.p[5::6]
        ],
        [None],
        [None],
        [IMGUI_YELLOW, IMGUI_RED, IMGUI_GREEN, IMGUI_BLUE],
        ['ths', 'd_ths', 'beta', 'd_beta'],
        'Planner Arguments',
    )
    plot_multiline(
        [speed_profile_tau],
        [planner.current_plan.l],
        [None],
        [None],
        [IMGUI_WHITE],
        ['l'],
        'Planner Point Distances',
    )

    if changed_1 or changed_2 or changed_3 or changed_4 or changed_5 or changed_6:
        ths = sim_state.ths
        s = sim_state.p.s
        y = sim_state.p.y
        for k in range(planner.config.N):
            planner_s[k] = s
            planner_y[k] = y
            planner_ths[k] = ths
            ths += kappa * delta / 2
            s += delta * np.cos(ths)
            y += delta * np.sin(ths)
            ths += kappa * delta / 2

        planner.solve(sim_state.vb.mag(), planner_s, planner_y, planner_ths)
        planner.current_plan.update_triangulated_plan(vis_plan, sim_surf)
        planner.current_plan.update_triangulated_plan_points(vis_plan_points, sim_surf)
        solve_times[:-1] = solve_times[1:]
        prep_times[:-1] = prep_times[1:]
        unpack_times[:-1] = unpack_times[1:]
        total_times[:-1] = total_times[1:]
        solve_times[-1] = planner.current_plan.solve_time * 1e6
        prep_times[-1] = planner.current_plan.prep_time * 1e6
        unpack_times[-1] = planner.current_plan.unpack_time * 1e6
        total_times[-1] = planner.current_plan.total_time * 1e6
        sim.zua2state(sim_state, *sim.state2zua(sim_state))
        sim.update_visual_assets(sim_state)

window.draw_vehicle_info = _window_extras
sim.update_visual_assets(sim_state)
while window.draw():
    pass
