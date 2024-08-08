'''
nonplanar racetrack model for the barc racetrack
a raceline is computed and a first-person animation is recorded
'''
import numpy as np
from numpy import pi

from vehicle_3d.surfaces.frenet_offset_surface import FrenetExpOffsetSurface,\
    FrenetExpOffsetSurfaceConfig
from vehicle_3d.models.tangent_vehicle_model import VehicleModelConfig
from vehicle_3d.raceline.kinematic_bicycle_raceline import KinematicRaceline, RacelineConfig
from vehicle_3d.visualization.window import Window

def _get_identified_surface(flip = True) -> FrenetExpOffsetSurface:
    # original s_max: 17.458270614976563

    config = FrenetExpOffsetSurfaceConfig(y_min = -0.4, y_max = 0.4, closed=True)

    ds = np.array(
        [2.251,pi*1.1515,0.901,1.0201675*pi/2,0.15,
        1.2281825*pi,2.25454,1.2206275*pi/2,0.905905])
    a = np.array([0,0,pi,pi,pi/2,pi/2,pi*1.5,pi*1.5,pi*2,pi*2]) - pi/2
    if flip:
        ds = np.flip(ds)
        a = np.flip(a)

    config.s = np.cumsum([0, *ds])
    config.a = a

    config.b = np.array([1.,1.]) * 6.1637
    config.s_b = np.array([0, config.s.max()])

    config.c = np.array([1., 1.]) * -0.3767
    config.s_c = np.array([0, config.s.max()])

    config.d = np.array([0., 0., 1., 1., 0., 0.]) * 0.092119
    config.s_d = np.array([0., 8.5, 9.3, 11.6, 12.5, config.s.max()])
    if flip:
        config.b *= -1
        config.c *= -1
        config.d = np.flip(config.d)
        config.s_d = config.s.max() - np.flip(config.s_d)

    config.e = np.array([1., 1.]) * 0.076
    config.s_e = np.array([0, config.s.max()])

    config.x0 = [-3.4669, 1.9382,0]

    surf = FrenetExpOffsetSurface(config)
    return surf

test_surf = _get_identified_surface()

solver = KinematicRaceline(
    test_surf,
    RacelineConfig(v_ws=1.0),
    VehicleModelConfig.barc_defaults())

raceline = solver.solve()

state = solver.model.get_empty_state()

T = np.linspace(raceline.t0, raceline.t0 + raceline.time, int(raceline.time * 60))

window = Window(test_surf)
window.show_imgui = False
window.camera_follow = True
window.camera_follow_q.from_vec([-1,1,1,-1])
window.camera_follow_q.normalize()
window.camera_follow_x.from_vec([0., 0., 0.])
window.start_recording(filename = 'barc_kinematic_raceline')

for t in T:
    solver.model.zu2state(
        state,
        raceline.z_interp(t),
        raceline.u_interp(t)
    )
    if not window.step(state):
        break
window.stop_recording()
window.close()
