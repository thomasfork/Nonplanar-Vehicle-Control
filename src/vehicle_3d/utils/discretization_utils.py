''' utilties for setting up and using collocation with casadi '''
import copy
import numpy as np
import casadi as ca

def get_collocation_coefficients(K, method = 'legendre'):
    '''
    return Kth decree collocation roots (tau)
    and coefficients for integral, derivative, and final state
    of collocation polynomial
    method can be 'legendre' or 'radau'
    '''
    tau = np.append(0, ca.collocation_points(K, method))

    B = np.zeros(K+1)      # collocation coefficients for integral
    C = np.zeros((K+1,K+1))# collocation coefficients for derivative
    D = np.zeros(K+1)      # collocation coefficients for continuity at end of interval (last value)

    for j in range(K+1):
        p = np.poly1d([1])
        for r in range(K+1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

        integral = np.polyint(p)
        B[j] = integral(1.0)

        tangent = np.polyder(p)
        for r in range(K+1):
            C[j,r] = tangent(tau[r])

        D[j] = p(1.0)
    return tau, B, C, D

def get_intermediate_collocation_coefficients(K, d, method = 'legendre'):
    '''
    return 'D' array (see above function) for a point d between 0 and 1
    '''
    tau = np.append(0, ca.collocation_points(K, method))

    D = np.zeros(K+1)

    for j in range(K+1):
        p = np.poly1d([1])
        for r in range(K+1):
            if r != j:
                p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

        D[j] = p(d)
    return D

def interpolate_collocation(w, H, X, config):
    '''
    generate an interpolation function for collocation
    '''

    # replace MX with equal size SX for building up interp. function
    w, H, X, _ = _replace_mx(w, H, X)

    t = ca.SX.sym('t')

    N = config.N
    K = config.K
    tau, _, _, D = get_collocation_coefficients(K)

    n = X[0,0].size()[0]

    tp = ca.cumsum(ca.vertcat(0,*H))

    pc = np.resize(np.array([], dtype = ca.SX), (K+1, n))
    for l in range(n):
        l0 = X[0,0][l]
        lf = 0
        for j in range(K+1):
            lf += X[-1,j][l] * D[j]

        for j in range(K+1):
            xp = ca.horzcat(*[l0, *[X[k,j][l] for k in range(N)], lf])
            pc[j,l] = ca.pw_const(t, tp, xp.T)

    t_offset = ca.pw_const(t, tp, ca.vertcat(0, tp))
    h = ca.pw_const(t, tp, ca.vertcat(1,*H,1))

    rel_t = (t - t_offset) / h
    p = [0] * n
    for l in range(n):
        for j in range(K+1):
            s = 1
            for r in range(K+1):
                if r != j:
                    s = s * (rel_t - tau[r]) / (tau[j] - tau[r])
            p[l] = p[l] + pc[j,l] * s

    f_collocate = ca.Function('x',[t,w], [ca.vertcat(*p)])

    return f_collocate

def interpolate_linear(w, H, X, xF = None):
    '''
    generate a linear interpolating function
    '''
    w, H, X, xF = _replace_mx(w, H, X, xF)

    tp = ca.cumsum(ca.vertcat(0, *H))
    xp = ca.horzcat(*X[:,0])
    t = ca.SX.sym('t')

    x_interp = [0] * xp.shape[0]
    for k in range(xp.shape[0]):
        if xF is not None:
            x_interp[k] = ca.pw_lin(t, tp, ca.vertcat(xp[k,:].T, xF[k]))
        else:
            x_interp[k] = ca.pw_lin(t, tp[:-1], xp[k,:].T)

    f_interp = ca.Function('x',[t,w], [ca.vertcat(*x_interp)])
    return f_interp

def interpolate_piecewise(w, H, X, xF = None):
    '''
    generate a piecewise constant interpolating function
    '''
    w, H, X, xF = _replace_mx(w, H, X, xF)

    tp = ca.cumsum(ca.vertcat(0, *H))
    xp = ca.horzcat(*X[:,0])
    t = ca.SX.sym('t')

    x_interp = [0] * xp.shape[0]
    for k in range(xp.shape[0]):
        if xF is not None:
            x_interp[k] = ca.pw_const(t, tp, ca.vertcat(0, xp[k,:].T, xF[k]))
        else:
            x_interp[k] = ca.pw_const(t, tp[:-1], ca.vertcat(0, xp[k,:].T))

    f_interp = ca.Function('x',[t,w], [ca.vertcat(*x_interp)])
    return f_interp

def _replace_mx(w, H, X, xF = None):
    '''
    replace MX expressions with SX of identical size
    for the purposes of building unpacking functions
    not meant for external use.
    '''
    if isinstance(X[0,0], ca.MX):
        w_sx = ca.SX.sym('w', w.shape)

        H = copy.deepcopy(H)
        X = copy.deepcopy(X)

        for n, h  in enumerate(H):
            if isinstance(h, ca.MX):
                extract_h = ca.Function('h', [w], [h])
                H[n] = extract_h(w_sx)

        for n in range(X.shape[0]):
            for k in range(X.shape[1]):
                if isinstance(X[n,k], ca.MX):
                    extract_x = ca.Function('h', [w], [X[n,k]])
                    X[n,k] = extract_x(w_sx)

        if isinstance(xF, ca.MX):
            extract_xf = ca.Function('xF', [w], [xF])
            xF = extract_xf(w_sx)

        w = w_sx

    return w, H, X, xF
