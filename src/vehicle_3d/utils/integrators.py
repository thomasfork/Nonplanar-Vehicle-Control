''' integrators for simulating dynamics models'''
from functools import singledispatch

import casadi as ca
import numpy as np

def idas_integrator(z, u, zdot, dt):
    ''' ode simulator based on SUNDIALS IDAS '''
    prob    = {'x':z, 'p':u, 'ode':zdot}
    try:
        # try integrator setup for casadi >= 3.6.0
        znewint = ca.integrator('zint','idas',prob, 0, dt)
    except NotImplementedError:
        setup   = {'t0':0, 'tf':dt}
        znewint = ca.integrator('zint','idas',prob, setup)

    if isinstance(z, ca.SX):
        z = ca.MX.sym('z', z.shape)
    if isinstance(u, ca.SX):
        u = ca.MX.sym('z', u.shape)
    znew    = znewint(x0=z,p=u)['xf']
    f_znew   = ca.Function('znew',[z,u],[znew],['z','u'],['znew'])

    def f(z, u):
        return np.array(f_znew(z, u)).squeeze()

    def f_ca(z, u):
        return f_znew(z, u)

    f = singledispatch(f)
    f.register(ca.SX, f_ca)
    f.register(ca.MX, f_ca)
    f.register(ca.DM, f_ca)

    return f

def idas_dae_integrator(z, u, a, zdot, h, dt):
    ''' dae simulator based on SUNDIALS IDAS '''
    prob    = {'x': z, 'p': u, 'z': a, 'ode': zdot, 'alg': h}
    try:
        # try integrator setup for casadi >= 3.6.0
        znewint = ca.integrator('zint','idas',prob, 0, dt)
    except NotImplementedError:
        setup   = {'t0':0, 'tf':dt}
        znewint = ca.integrator('zint','idas',prob, setup)

    if isinstance(z, ca.SX):
        z = ca.MX.sym('z', z.shape)
    if isinstance(u, ca.SX):
        u = ca.MX.sym('z', u.shape)
    a0 = ca.MX.sym('a0', a.shape)

    int_step    = znewint(x0=z,p=u,z0=a0)
    znew = int_step['xf']
    anew = int_step['zf']

    f_znew   = ca.Function('znew',[z,u,a0],[znew],['z','u','a0'],['znew'])
    f_zanew   = ca.Function('zanew',[z,u,a0],[znew, anew],['z','u','a0'],['znew', 'anew'])

    def f(z, u, a0):
        return np.array(f_znew(z, u, a0)).squeeze()

    def f_ca(z, u, a0):
        return f_znew(z, u, a0)

    f = singledispatch(f)
    f.register(ca.SX, f_ca)
    f.register(ca.MX, f_ca)
    f.register(ca.DM, f_ca)

    def f_za(z, u, a0):
        return (np.array(data).squeeze() for data in f_zanew(z, u, a0))

    def f_za_ca(z, u, a0):
        return f_zanew(z, u, a0)

    f_za = singledispatch(f_za)
    f_za.register(ca.SX, f_za_ca)
    f_za.register(ca.MX, f_za_ca)
    f_za.register(ca.DM, f_za_ca)

    return f, f_za
