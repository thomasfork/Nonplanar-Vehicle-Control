import numpy as np
import casadi as ca


def get_collocation_coefficients(K):
    tau = np.append(0, ca.collocation_points(K, 'legendre')) # legendre or radau

    B = np.zeros(K+1)      # collocation coefficients for quadrature (integral)
    C = np.zeros((K+1,K+1))# collocation coefficients for continuity within interval (derivative at each point)
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

def interpolate_collocation(w, H, X, config: 'RacelineConfig'):
    t = ca.SX.sym('t')
    
    N = config.N
    K = config.K
    tau, _, _, D = get_collocation_coefficients(K)
    
    n = X[0,0].size()[0]
    
    tp = ca.cumsum(ca.vertcat(0,H))
    
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
    h = ca.pw_const(t, tp, ca.vertcat(1,H,1))
    
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

