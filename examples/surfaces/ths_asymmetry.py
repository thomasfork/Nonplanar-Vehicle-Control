'''
illustration of asymmetry of theta^s for tangent road model
on a radially symmetric surface
'''

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.visualization.utils import get_instance_transforms
from vehicle_3d.visualization.gltf2 import load_car
from vehicle_3d.visualization.window import Window

surf = load_surface('hill')
r = 20
N = 1000
N_instances = 24
USE_TEX: bool = True
if USE_TEX:
    plt.rcParams['text.usetex'] = True

s = surf.sym_rep.s
y = surf.sym_rep.y

phi = ca.SX.sym('phi')

e2 = ca.horzcat(-ca.sin(phi), ca.cos(phi), 0)
eps = surf.p2eps(s, y)
epp = surf.p2epp(s, y)
ths = -ca.arctan2(e2 @ eps, e2 @ epp)

f_ths = surf.fill_in_param_terms(ths, [s, y, phi])

phi = np.linspace(0, 2*np.pi, N)
s = r * np.cos(phi + np.pi/3)
y = r * np.sin(phi + np.pi/3)
ths = f_ths(s[None], y[None], phi[None])
ths[ths < 0] += 2*np.pi

plt.plot(phi, ths)
if USE_TEX:
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\theta^s$')
else:
    plt.xlabel('Yaw Angle')
    plt.ylabel('Relative Heading Angle')
plt.tight_layout()
plt.show()

window = Window(surf)
car = load_car(window.ubo, color = [1,0,0,1], instanced = True)

phi = np.linspace(0, 2*np.pi, N_instances)
s = r * np.cos(phi + np.pi/3)
y = r * np.sin(phi + np.pi/3)
r = np.array([surf.pths2Rths(s[None], y[None], phi[None])])
r = r.T.reshape((-1,3,3))
r = r.transpose((0,2,1))
print(r.shape)
instances = get_instance_transforms(
    x = np.array([surf.p2x(s[None], y[None], 0.0)]).T.squeeze(),
    r = r,
)
car.apply_instancing(instances)
window.add_object('cars', car)
while window.draw():
    pass
