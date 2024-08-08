''' default tire model used for dynamic vehicle models '''
import casadi as ca
import numpy as np

from vehicle_3d.models.tire_model import TireConfig


def plot_tire_forces(config: TireConfig, linear:bool = False):
    ''' helper function to plot tire forces using matplotlib '''
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt
    from vehicle_3d.utils.ca_utils import ca_function_dispatcher
    s = ca.SX.sym('s')
    a = ca.SX.sym('a')
    Fx, Fy = config.combined_slip_forces(s, a, N=1, mu = 1, linear = linear)

    f_Fx = ca_function_dispatcher(ca.Function('Fx', [s,a], [Fx]))
    f_Fy = ca_function_dispatcher(ca.Function('Fy', [s,a], [Fy]))

    plt.figure()
    plt.subplot(2,1,1)
    s = np.linspace(-1, 1, 1000)
    A = np.linspace(0, 1, 11)
    for a in A:
        plt.plot(s, f_Fx(s[None], a), label = f'a = {a:0.2f}')
    plt.xlabel('Slip Ratio')
    plt.ylabel('Fx/(mu*N)')
    plt.legend()

    plt.subplot(2,1,2)
    a = np.linspace(-1, 1, 1000)
    S = np.linspace(0, 1, 11)
    for s in S:
        plt.plot(a, f_Fy(s, a[None]), label = f's = {s:0.2f}')
    plt.xlabel('Slip Angle')
    plt.ylabel('Fy/(mu*N)')
    plt.legend()

    plt.show()

plot_tire_forces(TireConfig())
