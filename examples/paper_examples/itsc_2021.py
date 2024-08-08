'''
main script for the paper

Fork, Thomas, H. Eric Tseng, and Francesco Borrelli.
"Models and predictive control for nonplanar vehicle
navigation." 2021 IEEE international intelligent
transportation systems conference (ITSC). IEEE, 2021.

Available Online: https://arxiv.org/pdf/2104.08427

Disclaimer: This example has been reproduced as faithfully as possible
for exact original code refer to
https://github.com/thomasfork/Nonplanar-Vehicle-Control/tree/2a7992c540ec365f3840d32679159e2a0f37df2e

Several traditional and model-based lane followers are tested and compared with 
simplified vehicle operating limit considerations. Later publications involve more
comprehensive operating limit considerations. 
'''
# pylint: disable=line-too-long
import timeit
import sys
from typing import Optional

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.models.tangent_vehicle_model import KinematicVehicleModel, VehicleModelConfig
from vehicle_3d.control.lane_following.stanley_pid_follower import SimpleStanleyPIDController, StanleyConfig
from vehicle_3d.control.lane_following.mpc_lane_follower import NonplanarMPC, NonplanarMPCConfig
from vehicle_3d.visualization.window import Window


def run_lap(controller,
            simulator,
            surf,
            lap = '',
            verbose = True,
            window: Optional[Window] = None):
    ''' basic utility for running a lap with a controller and plotting along the way'''
    state = simulator.get_empty_state()
    n = 0
    done = False
    traj = []
    avg_dt = 0
    solve_times = []

    while not done:
        ts = timeit.default_timer()
        t_sol = controller.step(state)
        solve_times.append(t_sol)

        simulator.step(state)

        if state.p.s > surf.s_max()-5:
            done = True

        traj.append(state.copy())

        tf = timeit.default_timer()
        avg_dt = n/(n+1) * avg_dt + 1/(n+1) * ((tf-ts))

        if window is not None:
            simulator.update_visual_assets(state)
            window.step(state)

        if verbose:
            if n > 0:
                sys.stdout.write("\033[K") # clear line
                sys.stdout.write("\033[K") # clear line
                sys.stdout.write("\033[K") # clear line
            n_tot = state.tfr.N + state.tfl.N + state.trr.N + state.trl.N
            print(f'Lap: {lap}   Contact: {("OK  " if n_tot >0 else "LOST")}')
            print(f'Vehicle @ {state.p.s:8.2f}/{surf.s_max()-5:8.2f}')
            print(f'Running @ {1/(tf-ts):8.2f}Hz (avg: {1/avg_dt:8.2f}Hz)')
            sys.stdout.write("\033[F") # Cursor up one line
            sys.stdout.write("\033[F") # Cursor up one line
            sys.stdout.write("\033[F") # Cursor up one line

        n += 1
    return traj, solve_times

def plot_solve_time(traj, solve_times):
    ''' plot controller solve time for a trajectory '''
    s  = [s.p.s for s in traj]
    plt.plot(s,solve_times)
    plt.xlabel(r'$s$')
    plt.ylabel('Solve Time (ms)')

def plot_timeseries_results(trajs, styles, vehicle_config, mpc):
    ''' plot general results of a closed loop trajectory'''
    plt.figure()
    gs1 = gridspec.GridSpec(7, 1)
    gs1.update(wspace=0.025, hspace=0.1) # set the spacing between axes.

    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])
    ax3 = plt.subplot(gs1[2])
    ax4 = plt.subplot(gs1[3])
    ax5 = plt.subplot(gs1[4])
    ax6 = plt.subplot(gs1[5])
    ax7 = plt.subplot(gs1[6])
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])
    ax4.set_xticklabels([])
    ax5.set_xticklabels([])
    ax6.set_xticklabels([])

    for (traj,style) in zip(trajs, styles):
        s = [s.p.s   for s in traj]
        y = [s.p.y   for s in traj]
        th = [s.ths for s in traj]
        tha = [np.arctan(vehicle_config.lr / (vehicle_config.lf + vehicle_config.lr) * np.tan(s.u.y)) + s.ths for s in traj]
        v = [s.vb.mag()   for s in traj]
        ua = [float(s.u.a - s.u.b)   for s in traj]
        uy = [float(s.u.y)   for s in traj]
        N = [float(s.tfr.N + s.tfl.N + s.trr.N + s.trl.N) for s in traj]

        ax1.plot(s,y, style)
        ax2.plot(s,th, style)
        ax3.plot(s,tha,style)
        ax4.plot(s,v,  style)
        ax5.plot(s,N,  style)
        ax6.plot(s,ua,  style)
        ax7.plot(s,uy,  style)

    ax5.fill_between([min(s), max(s)], [mpc.config.Nmax, mpc.config.Nmax], [mpc.config.Nmin, mpc.config.Nmin], color = 'green',alpha = 0.3)
    ax5.plot([min(s), max(s)], [0, 0], 'k')

    ax1.set_ylabel(r'$y$')
    ax2.set_ylabel(r'$\theta^s$')
    ax3.set_ylabel(r'$\theta^s + \beta$')
    ax4.set_ylabel(r'$v$')
    ax5.set_ylabel(r'$F_N^b$')
    ax6.set_ylabel(r'$a_t$')
    ax7.set_ylabel(r'$\gamma$')
    ax7.set_xlabel(r'$s$')

    ax1.yaxis.tick_right()
    ax2.yaxis.tick_right()
    ax3.yaxis.tick_right()
    ax4.yaxis.tick_right()
    ax5.yaxis.tick_right()
    ax6.yaxis.tick_right()
    ax7.yaxis.tick_right()
    plt.tight_layout()

def _main():
    vref = 10
    yref = 0
    dt = 0.05
    surf = load_surface('track_with_loop')

    vehicle_config = VehicleModelConfig(
        dt = dt,
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

    simulator = KinematicVehicleModel(config = vehicle_config, surf = surf)

    nonplanar_model = simulator
    planar_model = KinematicVehicleModel(config = planar_vehicle_config, surf = surf)

    stanley_controller  = SimpleStanleyPIDController(                    StanleyConfig(dt = vehicle_config.dt, vref = vref,yref = yref))
    mpc                 = NonplanarMPC(model = nonplanar_model, config = NonplanarMPCConfig(dt = vehicle_config.dt, use_planar_setup = False, vref = vref,yref = yref))
    pmpc                = NonplanarMPC(model = planar_model,    config = NonplanarMPCConfig(dt = vehicle_config.dt, use_planar_setup = True,  vref = vref,yref = yref))

    window = Window(surf)
    for label, item in simulator.generate_visual_assets(window.ubo).items():
        window.add_object(label, item)
    window.camera_follow=True

    stanley_traj, _ = run_lap(stanley_controller, simulator, surf, lap = 'stanley', window = window)
    pmpc_traj, pmpc_solve_times    = run_lap(pmpc,               simulator, surf, lap = 'planar mpc', window = window)
    mpc_traj, mpc_solve_times     = run_lap(mpc,                simulator, surf, lap = 'nonplanar mpc', window = window)

    window.close()

    print(f'Avg planar mpc solve time: {np.mean(pmpc_solve_times):0.3f}')
    print(f'Avg nonplanar mpc solve time: {np.mean(mpc_solve_times):0.3f}')

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams.update({'font.size': 18})
    #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    plot_timeseries_results([stanley_traj, pmpc_traj, mpc_traj], [':r', '--g', 'b'], vehicle_config, mpc)

    plt.figure()
    plot_solve_time(pmpc_traj, pmpc_solve_times)
    plot_solve_time(mpc_traj, mpc_solve_times)
    plt.legend(('Planar','Nonplanar'))
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    _main()
