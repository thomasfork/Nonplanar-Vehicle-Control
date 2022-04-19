''' integrators for simulating dynamics models'''
import numpy as np
import casadi as ca

from barc3d.utils.collocation_utils import get_collocation_coefficients

def collocation_integrator(z, u, f_zdot, dt, K = 7):
    ''' ode simulator based on collocation '''
    tau, B, C, D = get_collocation_coefficients(K)
        
    h = dt
        
    w = []       # states
    g = []       # nonlinear constraint functions
    lbg = []     # lower bound on constraint functions
    ubg = []     # upper bound on constraint functions
        
    state_dim = f_zdot.size_in(0)
    
    # create variables
    Z = np.resize(np.array([], dtype = ca.SX), (K+1))
    for j in range(0, K+1):
        zkj = ca.MX.sym('z_%d'%(j), state_dim)
        Z[j] = zkj
        w += [zkj]
    
    # initial condition        
    g += [z - Z[0]]
    ubg += [0.] * state_dim[0]
    lbg += [0.] * state_dim[0]
    
    # ODE
    for j in range(1,K+1):
        poly_ode = 0
        for j2 in range(K+1):
            poly_ode += C[j2][j] * Z[j2] / h

        func_ode = f_zdot(Z[j], u)
            
        g += [func_ode - poly_ode]
        ubg += [0.] * state_dim[0]
        lbg += [0.] * state_dim[0]
    
    # final state
    zN = 0
    for j in range(K+1):
        zN += D[j] * Z[j]
    
    # set up as an optimization problem
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    p = ca.vertcat(z,u)
        
    prob = {'f':0,'x':w, 'p': p, 'g': g}
    opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)
    
    # package with final state as output
    f_zN = ca.Function('zN',[w],[zN])
        
    zmx = ca.MX.sym('z', z.size())
    umx = ca.MX.sym('u', u.size())
    pmx = ca.vertcat(zmx, umx)
    sym_sol = solver.call([0,pmx,-np.inf, np.inf,lbg, ubg,0,0])[0]
    zN = f_zN.call([sym_sol])
        
    f_znew = ca.Function('zN', [zmx, umx], zN)
        
    return f_znew
    

def collocation_dae_integrator(z, u, g, f_zdot, f_gc, dt, vehicle_config, K = 7):
    ''' dae simulator based on collocation '''
    tau, B, C, D = get_collocation_coefficients(K)
        
    h = dt
    g0 = vehicle_config.m * vehicle_config.g
        
    w = []       # states
    g = []       # nonlinear constraint functions
    lbg = []     # lower bound on constraint functions
    ubg = []     # upper bound on constraint functions
        
    state_dim = f_zdot.size_in(0)
    alg_dim   = f_zdot.size_in(2)
    dae_dim   = f_gc.size_out(0)
    
    # create variables
    Z = np.resize(np.array([], dtype = ca.MX), (K+1))
    G = np.resize(np.array([], dtype = ca.MX), (K+1))
    for j in range(0, K+1):
        zkj = ca.MX.sym('z_%d'%(j), state_dim)
        Z[j] = zkj
        w += [zkj]
        
        gkj = ca.MX.sym('g_%d'%(j), alg_dim)
        G[j] = gkj
        w += [gkj]
    
    # initial condition
    g += [z - Z[0]]
    ubg += [0.] * state_dim[0]
    lbg += [0.] * state_dim[0]
    
    # DAE
    for j in range(0,K+1):
        poly_ode = 0
        for j2 in range(K+1):
            poly_ode += C[j2][j] * Z[j2] / h

        func_ode = f_zdot(Z[j], u, G[j]*g0)
        
        if j > 0:
            g += [func_ode - poly_ode]
            ubg += [0.] * state_dim[0]
            lbg += [0.] * state_dim[0]
        
        g += [f_gc(Z[j], u, G[j]*g0)]
        ubg += [0.] * dae_dim[0]
        lbg += [0.] * dae_dim[0]
    
    # final differential state
    zN = 0
    for j in range(K+1):
        zN += D[j] * Z[j]
    
    # final algebraic state
    gN = ca.MX.sym('g_N', alg_dim)
    w += [gN]
    g += [f_gc(Z[j], u, gN*g0)]
    ubg += [0.] * dae_dim[0]
    lbg += [0.] * dae_dim[0]
    
    # set up as an optimization problem
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    p = ca.vertcat(z,u)
    
    prob = {'f':0,'x':w, 'p': p, 'g': g}
    opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts)
    
    # package with final state as output (diff. only or diff. and alg states)
    f_zN = ca.Function('zN',[w],[zN])
    f_zgN = ca.Function('zgN',[w],[zN, gN*g0])
        
    zmx = ca.MX.sym('z', z.size())
    umx = ca.MX.sym('u', u.size())
    pmx = ca.vertcat(zmx, umx)
    sym_sol = solver.call([0,pmx,-np.inf, np.inf,lbg, ubg,0,0])[0]
    zN = f_zN.call([sym_sol])
    zgN = f_zgN.call([sym_sol])
        
    f_znew = ca.Function('zN', [zmx, umx], zN)
    f_zgnew = ca.Function('zN', [zmx, umx], zgN)
    return f_znew, f_zgnew
    

def idas_integrator(zmx, umx, zdot, dt):
    ''' ode simulator based on SUNDIALS IDAS '''
    prob    = {'x':zmx, 'p':umx, 'ode':zdot}
    setup   = {'t0':0, 'tf':dt}
    znewint = ca.integrator('zint','idas',prob, setup) 
    znew    = znewint.call([zmx,umx,0,0,0,0])[0]
    f_znew   = ca.Function('znew',[zmx,umx],[znew],['z','u'],['znew'])
    return f_znew

def idas_dae_integrator(zmx, umx, gmx, zdot, gc, dt):
    ''' dae simulator based on SUNDIALS IDAS '''
    prob    = {'x':zmx, 'p':umx, 'z':gmx, 'ode':zdot, 'alg':gc}
    setup   = {'t0':0, 'tf':dt}
    znewint = ca.integrator('zint','idas',prob, setup) 
    new_res = znewint.call([zmx,umx,0,0,0,0])
        
    znew = new_res[0]
    zgnew = new_res[0:3:2]
        
    f_znew   = ca.Function('znew',[zmx,umx],[znew],['z','u'],['znew'])
    f_zgnew   = ca.Function('znew',[zmx,umx],zgnew,['z','u'],['znew','gnew'])
    return f_znew, f_zgnew
    
