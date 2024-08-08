'''
planner for computing speed profiles on a nonplanar surface
'''
from dataclasses import dataclass, field
from typing import Callable
import timeit

import casadi as ca
import numpy as np
try:
    import ecos
except ImportError as e:
    raise RuntimeError('ECOS solver must be installed for speed planner') from e

from vehicle_3d.pytypes import PythonMsg, BaseTangentBodyState
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.utils.ca_utils import ca_function_dispatcher, ca_grad, coerce_angle, \
    coerce_angle_diff

from vehicle_3d.visualization.utils import triangulate_trajectory, get_sphere, \
    get_instance_transforms
from vehicle_3d.visualization.objects import VertexObject, UBOObject, InstancedVertexObject

from vehicle_3d.models.vehicle_model import VehicleModelConfig

ECOS_PASS_FLAGS = [0, 10] # 0 - solved and optimal, 10 = close to optimal
ECOS_FAIL_WARNINGS = {
    1: None, # primal infeasible
    2: None, # dual infeasible
    -2: 'ECOS Solver ran into numerical issues, this is probably a numerical conditioning issue'
}

@dataclass
class SpeedPlannerConfig(PythonMsg):
    ''' configuration for speed planner '''
    n: float = field(default = 0.5)
    N: int = field(default = 25)
    mu_limit: float = field(default = 0.5)
    v_max: float = field(default = 100)

    full_unpack: bool = field(default = True)
    yaw_input: bool = field(default = False)
    ''' changes the usual "ths" input to a global frame yaw angle target '''

@dataclass
class SpeedPlan(PythonMsg):
    '''
    outputs from speed planner
    interpolants are a function of [0,1]
    '''
    feasible: bool = field(default = None)
    prep_time: float = field(default = -1.)
    solve_time: float = field(default = -1.)
    solver_time: float = field(default = -1.)
    unpack_time: float = field(default = -1.)
    total_time: float = field(default = -1.)
    p: np.ndarray = field(default = None)
    l: np.ndarray = field(default = None)
    n: float = field(default = 0.)
    v: np.ndarray = field(default = None)
    v_max: float = field(default = 0.)
    v_min: float = field(default = 0.)
    N: np.ndarray = field(default = None)
    min_N: float = field(default = 0.)
    mu: np.ndarray = field(default = None)
    max_mu: float = field(default = 0.)
    s_interp: Callable[[float], float] = field(default = None)
    y_interp: Callable[[float], float] = field(default = None)
    v_interp: Callable[[float], float] = field(default = None)

    def triangulate_plan(self,
            ubo: UBOObject,
            surf: BaseSurface,
            v_max: float = None,
            v_min: float = None,
            n: int = 1000) -> VertexObject:
        '''
        triangulate the plan, returning a 3D line along it, colored by speed
        v_min and v_max provide optional different colorbar limits
        n adjust the number of points in the trajectory
        '''
        V, I = self._VI_trajectory(surf, v_max, v_min, n)
        return VertexObject(ubo, V, I, simple=True)

    def update_triangulated_plan(self,
            obj: VertexObject,
            surf: BaseSurface,
            v_max: float = None,
            v_min: float = None,
            n: int = 1000) -> None:
        '''
        update a triangulated plan object that has already been created,
        such as if the plan has been recomputed
        '''
        obj.setup(*self._VI_trajectory(surf, v_max, v_min, n))

    def _VI_trajectory(self,
            surf: BaseSurface,
            v_max: float = None,
            v_min: float = None,
            n: int = 1000,
            ):

        t = np.linspace(0, 1, n)
        s = self.s_interp(t)[None]
        y = self.y_interp(t)[None]
        v = self.v_interp(t)
        x = surf.p2x_fast(s, y, self.n).T
        e2 = surf.p2epp(s, y).T
        e3 = surf.p2xpn(s, y).T

        V, I, _, _ = triangulate_trajectory(
            x = x,
            e2 = e2,
            e3 = e3,
            v = v,
            v_max = v_max,
            v_min = v_min,
            closed = False,
            n = n
        )
        return V, I

    def triangulate_plan_points(self,
            ubo: UBOObject,
            surf: BaseSurface,
            ) -> InstancedVertexObject:
        ''' get a triangulation of the plan with spheres at each point along the plan '''
        V, I = get_sphere(r = 0.3)
        obj = InstancedVertexObject(ubo, V, I)
        self.update_triangulated_plan_points(obj, surf)
        return obj

    def update_triangulated_plan_points(self,
            obj: InstancedVertexObject,
            surf: BaseSurface):
        ''' update point triangulation '''
        obj.apply_instancing(
            self._trajectory_instances(surf)
        )

    def _trajectory_instances(self, surf: BaseSurface):
        x = surf.p2x_fast(self.p[::6][None], self.p[1::6][None], self.n).T
        return get_instance_transforms(x)


class SpeedPlanner:
    ''' speed planner '''
    surf: BaseSurface
    config: SpeedPlannerConfig
    vehicle_config: VehicleModelConfig
    current_plan: SpeedPlan = None

    _preproccess_function: Callable

    def __init__(self,
            surf: BaseSurface,
            vehicle_config: VehicleModelConfig,
            config: SpeedPlannerConfig = None):
        self.surf = surf
        self.config = config
        self.vehicle_config = vehicle_config
        if self.config is None:
            self.config = SpeedPlannerConfig()
        self._create_ocp()
        self._create_plan_preprocess_fn()

    def _stage_limit_fn(self):
        vsq = ca.SX.sym('vsq')
        v_dot = ca.SX.sym('v_dot')

        ths = self.surf.sym_rep.ths
        beta = ca.SX.sym('beta')
        d_ths = ca.SX.sym('d_ths')
        d_beta = ca.SX.sym('d_beta')

        one = self.surf.sym_rep.one
        two = self.surf.sym_rep.two
        n = self.config.n
        Q = self.surf.sym_rep.Q
        Q_inv = self.surf.sym_rep.Q_inv
        ks = self.surf.sym_rep.ws
        ky = self.surf.sym_rep.wy

        v_frac = ca.vertcat(ca.cos(ths+beta), ca.sin(ths+beta))
        p2v = ca.inv(one - n * two) @ Q @ v_frac
        wb3_per_v = d_ths - ca.horzcat(ks, ky) @ p2v
        wb12_per_v = self.surf.sym_rep.J_inv @ two @ p2v
        wb2_per_v = -wb12_per_v[0]
        wb1_per_v = wb12_per_v[1]

        Fb1 = self.vehicle_config.m * (
            v_dot * ca.cos(beta)
            - vsq * ca.sin(beta) * d_beta
            - vsq * ca.sin(beta) * wb3_per_v
        )
        Fb2 = self.vehicle_config.m * (
            v_dot * ca.sin(beta)
            + vsq * ca.cos(beta) * d_beta
            + vsq * ca.cos(beta) * wb3_per_v
        )
        Fb3 = self.vehicle_config.m * vsq * (
            v_frac.T @ Q_inv @ two @ p2v
        )

        R = self.surf.sym_rep.R_ths
        e1 = R[:,0]
        e2 = R[:,1]
        e3 = R[:,2]
        Fg1 = -self.vehicle_config.m * self.vehicle_config.g * e1[2]
        Fg2 = -self.vehicle_config.m * self.vehicle_config.g * e2[2]
        Fg3 = -self.vehicle_config.m * self.vehicle_config.g * e3[2]

        wb1_dot = wb1_per_v * v_dot
        wb2_dot = wb2_per_v * v_dot

        #NOTE - aerodynamic forces and moments ignored here
        Ft1 = Fb1 - Fg1
        Ft2 = Fb2 - Fg2
        FN3 = Fb3 - Fg3
        KN1 = self.vehicle_config.I1 * wb1_dot \
            + (self.vehicle_config.I3 - self.vehicle_config.I2) * wb2_per_v*wb3_per_v * vsq \
            - n * Ft2
        KN2 = self.vehicle_config.I2 * wb2_dot \
            + (self.vehicle_config.I1 - self.vehicle_config.I3) * wb1_per_v*wb3_per_v * vsq \
            + n * Ft1

        Nf = (FN3 * self.vehicle_config.lr - KN2) / self.vehicle_config.L
        Nr = (FN3 * self.vehicle_config.lr + KN2) / self.vehicle_config.L
        Delta = KN1 / 2 / (self.vehicle_config.tf**2 + self.vehicle_config.tr**2)

        Nfr = Nf/2 - self.vehicle_config.tf * Delta
        Nfl = Nf/2 + self.vehicle_config.tf * Delta
        Nrr = Nr/2 - self.vehicle_config.tr * Delta
        Nrl = Nr/2 + self.vehicle_config.tr * Delta

        normal_forces = ca.vertcat(Nfr, Nfl, Nrr, Nrl)

        z = ca.vertcat(vsq, v_dot)
        p = ca.vertcat(self.surf.sym_rep.s, self.surf.sym_rep.y, ths, d_ths, beta, d_beta)

        f_Ft = self.surf.fill_in_param_terms(
            ca.vertcat(Ft1, Ft2, FN3),
            [z, p]
        )
        f_N = self.surf.fill_in_param_terms(
            normal_forces,
            [z, p]
        )
        return f_Ft, f_N

    def _stage_continuity_fn(self):
        z0 = ca.SX.sym('z0', 2)
        z1 = ca.SX.sym('z1', 2)
        p0 = ca.SX.sym('p0', 6)
        p1 = ca.SX.sym('p1', 6)
        l0 = ca.SX.sym('l0')
        l1 = ca.SX.sym('l1')

        g = (z1[0] - z0[0]) - (z1[1] + z0[1]) * (l1 - l0)

        f_g = self.surf.fill_in_param_terms(
            g,
            [z0, p0, l0, z1, p1, l1]
        )
        return f_g

    def _create_ocp(self):
        J = 0
        x = []
        A = []
        b = []
        G_orth = []
        h_orth = []
        G_conic = []
        h_conic = []
        conic_dims = []
        z0 = []
        P = []
        L = []

        s_out = []
        y_out = []
        v_out = []
        v_dot_out = []
        N_out = []
        mu_out = []

        force_reg = self.vehicle_config.m * self.vehicle_config.g

        f_g = self._stage_continuity_fn()
        f_Ft, f_N = self._stage_limit_fn()
        for k in range(self.config.N):
            # variables
            z = ca.SX.sym(f'z_{k}', 2)
            p = ca.SX.sym(f'p_{k}', 6)
            l = ca.SX.sym(f'l_{k}', 1)
            x += [z]
            P += [p]
            L += [l]
            s_out += [p[0]]
            y_out += [p[1]]
            v_out += [ca.sqrt(ca.if_else(z[0]>0, z[0], 0))]
            v_dot_out += [z[1]]

            # maximize speed
            J -= z[0]

            # continuity
            if k == 0:
                v0 = ca.SX.sym('v0')
                z0 += [v0]
                A += [z[0]]
                b += [v0**2]
            else:
                A += [f_g(x[-2], P[-2], L[-2], x[-1], P[-1], L[-1])]
                b += [0.]

            # stage constraints
            Ft = f_Ft(z, p)
            N = f_N(z, p)
            N_out += [N]

            # positive orthant constraints
            # min and max speed
            G_orth += [z[0], -z[0]]
            h_orth += [self.config.v_max**2, 0]
            # normal force
            G_orth += [-N / force_reg]
            h_orth += [ca.substitute(N, z, (0,0)) / force_reg]

            # second order cone constraints
            # friction limit
            soc = ca.vertcat(
                -self.config.mu_limit * Ft[2] / force_reg,
                Ft[0] / force_reg,
                Ft[1] / force_reg
            )
            G_conic += [
                soc
            ]
            h_conic += [-ca.substitute(soc, z, (0,0))]
            conic_dims += [3]
            mu_out += [
                ca.norm_2(ca.vertcat(Ft[0], Ft[1])) \
                / Ft[2]
            ]


        x = ca.vertcat(*x)
        p = ca.vertcat(*P)
        l = ca.vertcat(*L)
        z0 = ca.vertcat(*z0)

        v = ca.vertcat(*v_out)
        v_dot = ca.vertcat(*v_dot_out)
        N = ca.horzcat(*N_out).T
        mu = ca.vertcat(*mu_out)

        A = ca.vertcat(*A)
        b = ca.vertcat(*b)

        G_orth = ca.vertcat(*G_orth)
        h_orth = ca.vertcat(*h_orth)
        G_conic = ca.vertcat(*G_conic)
        h_conic = ca.vertcat(*h_conic)

        c = ca.jacobian(J, x).T
        G = ca.jacobian(ca.vertcat(G_orth, G_conic), x)
        h = ca.vertcat(h_orth, h_conic)
        A = ca.jacobian(A, x)

        self._ecos_dims = {
            'l': G_orth.shape[0],
            'q': conic_dims,
        }

        self._f_ecos_args = ca.Function('ecos_args', [z0, p, l], [c, G, h, A, b])
        self._f_ecos_outs = ca_function_dispatcher(
            ca.Function('v', [x, z0, p, l], [v, v_dot, N, mu]))

    def _create_plan_preprocess_fn(self):
        ds = ca.SX.sym('ds')
        dy = ca.SX.sym('dy')
        ths = self.surf.sym_rep.ths
        stage_args = [
            self.surf.sym_rep.s,
            self.surf.sym_rep.y,
            self.surf.sym_rep.n,
            ds,
            dy
        ]

        # differential line segment length
        dsdy = ca.vertcat(ds,dy)
        metric = self.surf.sym_rep.one - self.config.n * self.surf.sym_rep.two
        dl = ca.sqrt(ca.bilin(metric, dsdy, dsdy))
        f_dl = self.surf.fill_in_param_terms(dl,
            stage_args
        )

        # sideslip angle
        vb12 = self.surf.sym_rep.Q_inv @ metric @ ca.vertcat(ds, dy)
        beta = ca.atan2(vb12[1], vb12[0]) - ths
        f_beta = self.surf.fill_in_param_terms(
            [beta],
            stage_args + [ths]
        )

        # plan arguments
        t = ca.SX.sym('t')
        tau = np.linspace(0, 1, self.config.N)
        S = ca.SX.sym('S', self.config.N)
        Y = ca.SX.sym('Y', self.config.N)
        s = ca.pw_lin(t, tau, S)
        y = ca.pw_lin(t, tau, Y)

        # integrator to estimate length of segments between stages
        ds = ca.jacobian(s, t)
        dy = ca.jacobian(y, t)
        dl = f_dl(s, y, self.config.n, ds, dy)
        config = {
            'output_t0':True,
            'grid':np.linspace(0,1,self.config.N),
            'max_step_size':(1/self.config.N)}
        ode = {
            'x': t,
            'p': ca.vertcat(S, Y),
            'ode': 1,
            'quad': dl,
        }
        try:
            # try integrator setup for casadi >= 3.6.0
            dl_int = ca.integrator('x','idas',ode, 0,
                config['grid'],
                {'max_step_size': (0.5/self.config.N)}
            )
        except NotImplementedError:
            dl_int = ca.integrator('x','idas',ode, config)

        # repackage everything using ca.MX since this is required for integrator call
        S_mx = ca.MX.sym('S', self.config.N)
        Y_mx = ca.MX.sym('Y', self.config.N)
        THS_mx = ca.MX.sym('THS', self.config.N)
        if self.config.yaw_input:
            e2 = ca.horzcat(-ca.sin(THS_mx), ca.cos(THS_mx), np.zeros(self.config.N))
            eps = self.surf.p2eps(S_mx.T, Y_mx.T).T
            epp = self.surf.p2epp(S_mx.T, Y_mx.T).T
            THS_COMP_mx = -ca.arctan2(
                ca.sum2(e2 * eps),
                ca.sum2(e2 * epp)
            )
            for k in range(1,self.config.N):
                THS_COMP_mx[k] = coerce_angle_diff(THS_COMP_mx[k], THS_COMP_mx[k-1])
        else:
            THS_COMP_mx = THS_mx
        L_mx = dl_int(x0 = 0, p = ca.vertcat(S_mx, Y_mx))['qf']
        DS_mx = ca_grad(S_mx, L_mx)
        DY_mx = ca_grad(Y_mx, L_mx)
        DTHS_mx = ca_grad(THS_COMP_mx, L_mx)
        BETA_mx = f_beta(S_mx, Y_mx, self.config.n, DS_mx, DY_mx, THS_COMP_mx)
        BETA_mx = coerce_angle(BETA_mx) # fix angle wraparound for beta
        DBETA_mx = ca_grad(BETA_mx, L_mx)

        P_mx = ca.horzcat(
            S_mx,
            Y_mx,
            THS_COMP_mx,
            DTHS_mx,
            BETA_mx,
            DBETA_mx
        ).T.reshape((-1,1))
        self._preproccess_function = self.surf.fill_in_param_terms(
            [P_mx, L_mx],
            [S_mx, Y_mx, THS_mx]
        )

    def _ecos_args(self, z0, p, l) -> dict:
        c,G,h,A,b = self._f_ecos_args(z0, p, l)
        c = np.array(c).squeeze()
        G = G.tocsc()
        h = np.array(h).squeeze()
        A = A.tocsc()
        b = np.array(b).squeeze()
        return {
            'c': c,
            'G': G,
            'h': h,
            'A': A,
            'b': b,
            'dims': self._ecos_dims,
            'verbose': False
        }

    def solve(self,
              v0: float,
              s: np.ndarray,
              y: np.ndarray,
              ths: np.ndarray):
        '''
        low level solve interface with all waypoints given
        vdot0 is currently ignored
        '''
        t0 = timeit.default_timer()

        # prepare arguments
        z0 = np.array([v0])
        p, l = self._preproccess_function(s,y,ths)
        solver_args = self._ecos_args(z0, p, l)
        t1 = timeit.default_timer()

        # solve
        sol = ecos.solve(**solver_args)
        t2 = timeit.default_timer()

        # unpack
        self._unpack_sol(sol, z0, p, l, t0, t1, t2)

    def solve_state(self, state: BaseTangentBodyState, ds: float = 2):
        '''
        solve the planner for a given state
        with a lane-keeping type trajectory
        '''
        self.solve(
            state.vb.mag(),
            state.p.s + np.arange(self.config.N) * ds,
            state.p.y * np.ones(self.config.N),
            state.ths * np.ones(self.config.N),
        )

    def _unpack_sol(self, sol, z0, p, l,
            t0 = -1,
            t1 = -1,
            t2 = -1
        ):
        v, _, N, mu = self._f_ecos_outs(sol['x'], z0, p, l)

        exit_flag = sol['info']['exitFlag']
        feas_and_opt = exit_flag in ECOS_PASS_FLAGS
        if not feas_and_opt:
            if exit_flag in ECOS_FAIL_WARNINGS:
                msg = ECOS_FAIL_WARNINGS[exit_flag]
                if msg is not None:
                    print(f'WARNING: {msg}')
            else:
                print(f'Unhandled ECOS Exit Flag: {exit_flag}')
                print(sol['info'])
        t_sol = sol['info']['timing']['runtime']

        if self.config.full_unpack:
            t_grid = np.linspace(0, 1, self.config.N)
            s_grid = np.array(p[0::6]).squeeze()
            y_grid = np.array(p[1::6]).squeeze()
            v_grid = np.array(v).squeeze()
            def s_interp(t):
                return np.interp(t, t_grid, s_grid)
            def y_interp(t):
                return np.interp(t, t_grid, y_grid)
            def v_interp(t):
                return np.interp(t, t_grid, v_grid)
        else:
            s_interp = None
            y_interp = None
            v_interp = None
        t3 = timeit.default_timer()

        self.current_plan = SpeedPlan(
            feasible = feas_and_opt,
            prep_time = t1 - t0,
            solve_time = t2 - t1,
            solver_time = t_sol,
            unpack_time = t3 - t2,
            total_time = t3 - t0,
            p = p,
            l = l,
            n = self.config.n,
            v = v,
            v_max = v.max(),
            v_min = v.min(),
            N = N,
            min_N = N.min(),
            mu = mu,
            max_mu = mu.max(),
            s_interp = s_interp,
            y_interp = y_interp,
            v_interp = v_interp,
        )
