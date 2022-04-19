'''
Standard methods and features for nonplanar raceline computation and manipulation
'''
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import numpy as np
import casadi as ca
import scipy.interpolate
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection

from barc3d.pytypes import PythonMsg
from barc3d.utils.collocation_utils import get_collocation_coefficients, interpolate_collocation
from barc3d.utils.ca_utils import unpack_solution_helper
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.dynamics.dynamics_3d import DynamicsModel
from barc3d.pytypes import VehicleState, VehicleConfig
from barc3d.visualization.raceline_fig import RacelineWindow
from barc3d.dynamics.dynamics_3d import DynamicTwoTrackSlipInput3D

matplotlib.rcParams.update({'font.size': 18})

@dataclass
class RacelineConfig(PythonMsg):
    ''' configuration for a raceline solver '''
    #problem parameters
    K: int = field(default = 7)
    N: int = field(default = 100)
    R: np.ndarray  = field(default = 0.001) # if a float - multiplied by appropriate size identity matrix
    dR: np.ndarray = field(default = 0.001)
    closed: bool = field(default = False) # if True, constrain final and initial states

    y_sep_l: float = field(default = 0.0)  # tighten lane boundary to the left (usually y>0)
    y_sep_r: float = field(default = 0.0)  # tighten lane boundary to the right (usually y<0)

    verbose: bool = field(default = False)

    # warm start parameters
    v_ws: float = field(default = 10)

    # initial constraints, ignored if not provided
    y0: float = field(default = None)
    ths0: float = field(default = None)
    v0: float = field(default = None)


@dataclass
class RacelineResults(PythonMsg):
    ''' template for storing raceline results '''
    raceline: VehicleState = field(default = None)
    step_sizes: np.ndarray = field(default = np.array([]))
    time: float = field(default = -1)
    label: str = field(default = '')
    color: np.ndarray = field(default = np.array([0,0,0,1]))

    z_interp: ca.Function = field(default = None)
    u_interp: ca.Function = field(default = None)
    du_interp: ca.Function = field(default = None)
    g_interp: scipy.interpolate._interpolate.interp1d = field(default = None)

class BaseRaceline(ABC):
    ''' base raceline class '''
    is_dae = False
    z_interp_collocation = None
    u_interp_collocation = None
    du_interp_collocation = None

    def __init__(self, surf: BaseSurface,
                       config: RacelineConfig,
                       vehicle_config: VehicleConfig,
                       ws_raceline: RacelineResults = None,
                       setup = True):
        self.surf = surf
        self.model = self._get_model(vehicle_config, surf)
        self.config = config
        
        # if costs have been provided as a float, convert to matrices
        if not isinstance(self.config.R, np.ndarray):
            self.config.R = float(self.config.R) * np.eye(self.model.f_zdot.size_in(1)[0])
        if not isinstance(self.config.dR, np.ndarray):
            self.config.dR = float(self.config.dR) * np.eye(self.model.f_zdot.size_in(1)[0])

        # warm start raceline - up to setup() function to use or discard.
        self.ws_raceline = ws_raceline
        
        if setup:
            self.setup(s0 = None, sf = None)

    @abstractmethod
    def _get_model(self, vehicle_config: VehicleConfig, surf: BaseSurface) -> DynamicsModel:
        '''obtain a vehicle model for a particular raceline class'''
        return

    @abstractmethod
    def _get_raceline_color(self) -> list:
        ''' return a RGBA color for the racleine, ie. [1,0,0,1]'''
        return

    @abstractmethod
    def _get_raceline_label(self):
        ''' return a string label for the raceline'''
        return

    @abstractmethod
    def _initial_setup_checks(self):
        raise NotImplementedError('Cannot Instantiate Solver of Base Class')

    def check_valid_warmstart(self):
        ''' check if self.ws_raceline is a valid warmstart object'''
        if self.ws_raceline is None:
            return False
        else:
            valid_step_sizes = True
            if not isinstance(self.ws_raceline.step_sizes, np.ndarray):
                print('provided warm start setp sizes is not a numpy array')
                valid_step_sizes = False
            elif not self.ws_raceline.step_sizes.ndim == 1:
                print('provided warm start step sizes array is not 1 dimensional')
                valid_step_sizes = False
            elif not self.ws_raceline.step_sizes.shape[0] == self.config.N:
                print('provided warm start step sizes array is not the correct length (N)')
                valid_step_sizes = False
            elif not np.issubdtype(self.ws_raceline.step_sizes.dtype, np.number):
                print('provided warm start step sizes array is not numeric')
                valid_step_sizes = False

            valid_states = True
            if not len(self.ws_raceline.raceline) == self.config.N * (self.config.K + 1):
                print('warm start raceline is not correct length (N*(K+1))')
                valid_states = False
            else:
                for state in self.ws_raceline.raceline:
                    if not isinstance(state, VehicleState):
                        valid_states = False
                        print('Found a provided warm start state of incorrect type')

        use_raceline = valid_states and valid_step_sizes

        if self.config.verbose:
            if use_raceline:
                print('No warmstart errors found, using provided raceline')
            else:
                print('Warmstart errors detected, discarding provided raceline')
                self.ws_raceline = None

        return use_raceline

    def setup(self, s0 = None, sf = None):
        ''' set up the raceline problem '''
        t0 = time.time()
        J, w, w0, ubw, lbw, g, ubg, lbg, H, Z, U, dU, G = self._build_opt(s0 = s0, sf = sf)

        prob = {'f':J,'x':w, 'g':g}
        if self.config.verbose:
            opts = {'ipopt.sb':'yes'}
        else:
            opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}

        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # keep track of bounds
        self.solver_ubw = ubw
        self.solver_lbw = lbw
        self.solver_w0 = w0
        self.solver_ubg = ubg
        self.solver_lbg = lbg

        self._build_soln_unpackers(w, H, Z, U, dU, G)
        self.setup_time = time.time() - t0

    def solve(self) -> RacelineResults:
        ''' solve the raceline problem, return results'''
        t0 = time.time()
        sol = self.solver(lbg = self.solver_lbg, ubg = self.solver_ubg,
                          x0 = self.solver_w0, ubx = self.solver_ubw, lbx = self.solver_lbw)
        self.solve_time = time.time() - t0
        self.sol = sol

        if self.config.verbose:
            sol_h = self.f_sol_h(sol['x'])
            print('reached target in   %0.3f seconds'%(np.sum(sol_h)))
            print('and a total cost of %0.3f'%sol['f'])
            print('it took \t%0.3f seconds to set up the problem'%(self.setup_time))
            print('    and \t%0.3f seconds to solve it'%(self.solve_time))

        return self._unpack_soln(sol)

    def _build_opt(self, s0 = None, sf = None):
        self._initial_setup_checks()

        warmstart = self.check_valid_warmstart()

        if s0 is None: s0 = self.surf.s_min()
        if sf is None: sf = self.surf.s_max()

        # collocation and planning
        N = self.config.N
        K = self.config.K
        tau, B, C, D = get_collocation_coefficients(K)

        # cost terms
        R = self.config.R
        dR = self.config.dR

        # model functions
        f_ode = self.model.f_zdot_full
        f_param_terms = self.model.f_param_terms
        if self.is_dae:
            f_dae = self.model.f_gc_full

        # model dimensions
        state_dim = f_ode.size_in(0)
        input_dim = f_ode.size_in(1)
        if self.is_dae:
            alg_dim   = f_ode.size_in(2)
            dae_dim   = f_dae.size_out(0)

        # basic dimension checks
        assert R.shape == (input_dim[0], input_dim[0])
        assert dR.shape == (input_dim[0], input_dim[0])

        # state and input limits
        state_u = self.model.zu()
        state_u[1] = state_u[1] - self.config.y_sep_l
        state_l = self.model.zl()
        state_l[1] = state_l[1] + self.config.y_sep_r
        input_u = self.model.uu()
        input_l = self.model.ul()
        input_du = self.model.duu()
        input_dl = self.model.dul()

        # modify some limits for forwards-only motion in racelines discretized in space
        state_u[0] = np.inf
        state_l[0] =-np.inf
        state_u[2] = np.pi
        state_l[2] =-np.pi
        state_l[3] = 0

        # path length for each collocation interval
        delta_s_col = (sf - s0) / N
        h0 = delta_s_col / self.config.v_ws

        # variable normalization
        if self.is_dae:
            g0 = self.model.m * self.model.g
            self.g0 = g0
  
        # raceline NLP
        w = []       # states     (vehicle state and inputs)
        w0 = []      # initial guesses for states
        lbw = []     # lower bound on states
        ubw = []     # upper bound on states
        J = 0        # cost
        g = []       # nonlinear constraint functions
        lbg = []     # lower bound on constraint functions
        ubg = []     # upper bound on constraint functions

        # matrices to organize states 
        H = np.resize(np.array([], dtype = ca.SX), (N))
        Z = np.resize(np.array([], dtype = ca.SX), (N,K+1))
        U = np.resize(np.array([], dtype = ca.SX), (N,K+1))
        dU = np.resize(np.array([], dtype = ca.SX), (N,K+1))

        if self.is_dae:
            G = np.resize(np.array([], dtype = ca.SX), (N,K+1))

        # create step size variables
        for k in range(N):
            hk = ca.SX.sym('h_%d'%k)
            H[k] = hk
            w += [hk]
            ubw += [h0*40]
            lbw += [h0/40]

            if warmstart:
                w0 += [self.ws_raceline.step_sizes[k]]
            else:
                w0 += [h0]

            J += hk # penalize total time

        # create the problem
        for k in range(N):
            # add collocation point variables
            for j in range(0, K+1):
                zkj = ca.SX.sym('z_%d_%d'%(k,j), state_dim)
                Z[k,j] = zkj
                w += [zkj]
                lbw += state_l
                ubw += state_u

                ukj = ca.SX.sym('u_%d_%d'%(k,j), input_dim)
                U[k,j] = ukj
                w += [ukj]
                lbw += input_l
                ubw += input_u

                dukj = ca.SX.sym('du_%d_%d'%(k,j), input_dim)
                dU[k,j] = dukj
                w += [dukj]
                lbw += input_dl
                ubw += input_du

                if self.is_dae:
                    gkj = ca.SX.sym('g_%d_%d'%(k,j), alg_dim)
                    G[k,j] = gkj
                    w += [gkj]
                    lbw += [-np.inf] * dae_dim[0]
                    ubw += [np.inf] * dae_dim[0]

                if warmstart:
                    ws_state = self.ws_raceline.raceline[j + k * (self.config.K + 1)]
                    if self.is_dae:
                        z, u, g_temp = self.model.state2zug(ws_state)
                    else:
                        z, u = self.model.state2zu(ws_state)
                    du      = self.model.state2du(ws_state)

                    w0 += z
                    w0 += u
                    w0 += du
                    if self.is_dae:
                        w0 += g_temp
                else:
                    w_guess = [0.] * state_dim[0]
                    w_guess[0] = delta_s_col*(k + tau[j]) + s0
                    w_guess[3] = self.config.v_ws
                    w0 += w_guess

                    w0 += [0.] * input_dim[0]
                    w0 += [0.] * input_dim[0]
                    if self.is_dae:
                        w0 += [self.model.lr / self.model.L, self.model.lf / self.model.L, 0]

            # add ode (dae) continuity constraints within the interval
            for j in range(K+1):
                poly_ode = 0
                poly_du = 0
                for j2 in range(K+1):
                    poly_ode += C[j2][j] * Z[k,j2] / H[k]
                    poly_du  += C[j2][j] * U[k,j2] / H[k]

                s_pt = delta_s_col * (k + tau[j]) + s0
                param_terms = f_param_terms((s_pt,0,0))

                if self.is_dae:
                    func_ode = f_ode(Z[k,j], U[k,j], G[k,j] * g0, param_terms)
                else:
                    func_ode = f_ode(Z[k,j], U[k,j], param_terms)

                if j > 0 or (k == 0 and not self.config.closed): # avoid overconstraining points that are constrained by the previous interval
                    if j == 0:
                        # path length of the start of each interval is constrained by spacing constraints
                        g += [func_ode[1:] - poly_ode[1:]]
                        ubg += [0.] * (state_dim[0]-1)
                        lbg += [0.] * (state_dim[0]-1)
                    else:
                        g += [func_ode - poly_ode]
                        ubg += [0.] * state_dim[0]
                        lbg += [0.] * state_dim[0]

                g += [dU[k,j] - poly_du]
                ubg += [0.] * input_dim[0]
                lbg += [0.] * input_dim[0]

                if self.is_dae:
                    func_dae = f_dae(Z[k,j], U[k,j], G[k,j] * g0, param_terms)
                    g += [func_dae]
                    ubg += [0.] * dae_dim[0]
                    lbg += [0.] * dae_dim[0]

                # add model specific stage constraints, ie. normal force constraints, friction cone, etc...
                if self.is_dae:
                    g, ubg, lbg = self.model._add_model_stage_constraints(Z[k,j], U[k,j], G[k,j] * g0, g, lbg, ubg, param_terms = param_terms)
                else:
                    g, ubg, lbg = self.model._add_model_stage_constraints(Z[k,j], U[k,j], g, lbg, ubg, param_terms = param_terms)

            # add quadrature costs
            poly_int = 0
            for j in range(K+1):
                stage_cost = ca.bilin(R, U[k,j], U[k,j]) + ca.bilin(dR, dU[k,j], dU[k,j])
                poly_int += B[j] * stage_cost * H[k]
            J += poly_int

            # add state continuity constraints (not to algebraic states)
            if k >= 1:
                poly_prev = 0
                poly_prev_input = 0
                for j in range(K+1):
                    poly_prev += Z[k-1,j] * D[j]
                    poly_prev_input += U[k-1,j] * D[j]

                # path length of the start of each interval is constrained by spacing constraints
                g += [Z[k,0][1:] - poly_prev[1:]]
                ubg += [0.] * (state_dim[0]-1)
                lbg += [0.] * (state_dim[0]-1)

                g += [U[k,0] - poly_prev_input]
                ubg += [0.] * input_dim[0]
                lbg += [0.] * input_dim[0]

            # state at the end of the interval
            zN = 0
            for j in range(K+1):
                zN += Z[k,j] * D[j]

            # path length at the start of the interval
            g += [Z[k,0][0] - (delta_s_col * k+ s0)]
            ubg += [0.]
            lbg += [0.]

            # path length at the end of the interval
            g += [zN[0] - (delta_s_col * (k + 1) + s0)]
            ubg += [0.]
            lbg += [0.]

        if self.config.closed:
            # loop closure constraint
            g += [Z[0,0][1:] - zN[1:]]
            ubg += [0.] * (state_dim[0] - 1)
            lbg += [0.] * (state_dim[0] - 1)

            uN = sum(U[-1,j]*D[j] for j in range(K+1))
            g += [U[0,0] - uN]
            ubg += [0.] * input_dim[0]
            lbg += [0.] * input_dim[0]
        else:
            g, ubg, lbg = self._add_model_initial_constraints(Z[0,0], U[0,0], g, lbg, ubg)

        g = ca.vertcat(*g)
        w = ca.vertcat(*w)

        self.z_interp_collocation = interpolate_collocation(w, H, Z, self.config)
        self.u_interp_collocation = interpolate_collocation(w, H, U, self.config)
        self.du_interp_collocation = interpolate_collocation(w, H, dU, self.config)

        if self.is_dae:
            return J, w, w0, ubw, lbw, g, ubg, lbg, H, Z, U, dU, G
        else:
            return J, w, w0, ubw, lbw, g, ubg, lbg, H, Z, U, dU, []

    def _add_model_initial_constraints(self, z0, u0, g, lbg, ubg):
        if self.config.y0 is not None:
            g += [z0[1]]
            ubg += [self.config.y0]
            lbg += [self.config.y0]

        if self.config.ths0 is not None:
            g += [z0[2]]
            ubg += [self.config.ths0]
            lbg += [self.config.ths0]

        if self.config.v0 is not None:
            g += [z0[3]]
            ubg += [self.config.v0]
            lbg += [self.config.v0]

        return g, ubg, lbg

    def _build_soln_unpackers(self, w, H, Z, U, dU, G = []):
        self.f_sol_h  = unpack_solution_helper('h' , w, H.tolist())
        self.f_sol_z  = unpack_solution_helper('z' , w, [sx for sx in np.concatenate(Z)])
        self.f_sol_u  = unpack_solution_helper('u' , w, [sx for sx in np.concatenate(U)])
        self.f_sol_du = unpack_solution_helper('du', w, [sx for sx in np.concatenate(dU)])

        if self.is_dae:
            self.f_sol_g  = unpack_solution_helper('g' , w, [sx * self.g0 for sx in np.concatenate(G)])

    def _unpack_soln(self, sol):
        raceline = []

        sol_h = self.f_sol_h(sol['x'])

        tau, _, _, _ = get_collocation_coefficients(self.config.K)
        t0 = np.cumsum(np.array([0, *sol_h[:-1]]))
        sol_t = np.concatenate(np.outer(tau, sol_h).T + t0[None].T)

        if not self.is_dae:
            sol_z = self.f_sol_z(sol['x'])
            sol_u = self.f_sol_u(sol['x'])
            sol_du = self.f_sol_du(sol['x'])
            
            g_interp = None
            
            for t, z, u, du in zip(sol_t, sol_z, sol_u, sol_du):
                state = VehicleState()
                state.t = t
                self.model.zn2state(state, z, u)
                self.model.du2state(state, du)
                raceline.append(state)

        else:
            sol_z = self.f_sol_z(sol['x'])
            sol_u = self.f_sol_u(sol['x'])
            sol_du = self.f_sol_du(sol['x'])
            sol_g = self.f_sol_g(sol['x'])
            
            g_interp = scipy.interpolate.interp1d(sol_t, sol_g, axis = 0, fill_value = 'extrapolate')
            
            for t, z, u, du, g in zip(sol_t, sol_z, sol_u, sol_du, sol_g):
                state = VehicleState()
                state.t = t
                self.model.zngn2state(state, z, g, u)
                self.model.du2state(state, du)
                raceline.append(state)

        t = ca.SX.sym('t')
        z_interp = self.z_interp_collocation.call([t, sol['x']])
        z_interp = ca.Function('z_interp', [t], z_interp)
        z_interp_np = lambda t: np.array(z_interp(t)).squeeze()
        u_interp = self.u_interp_collocation.call([t, sol['x']])
        u_interp = ca.Function('u_interp', [t], u_interp)
        u_interp_np = lambda t: np.array(u_interp(t)).squeeze()
        
        du_interp = self.du_interp_collocation.call([t, sol['x']])
        du_interp = ca.Function('du_interp', [t], du_interp)
        du_interp_np = lambda t: np.array(du_interp(t)).squeeze()

        return RacelineResults(raceline = raceline, step_sizes = sol_h, time = np.sum(sol_h),
                               label = self._get_raceline_label(), color = self._get_raceline_color(),
                               z_interp = z_interp_np, u_interp = u_interp_np, du_interp = du_interp_np,
                               g_interp = g_interp)     

    def plot_raceline_pyplot(self, results, block = True):
        center = []
        left = []
        right = []
        
        for s in np.linspace(self.surf.s_min(), self.surf.s_max(), 1000):
            center.append(self.surf.ro2x((s,0,0)))
            left.append(self.surf.ro2x((s,self.surf.y_max(),0)))
            right.append(self.surf.ro2x((s,self.surf.y_min(),0)))
        center = np.array(center)
        left = np.array(left)
        right = np.array(right)

        fig = plt.figure()
        plt.plot(center[:,0], center[:,1],'--r')
        plt.plot(left[:,0], left[:,1],'-k')
        plt.plot(right[:,0], right[:,1],'-k')
        plt.xlabel('xi (m)')
        plt.ylabel('xj (m)')

        if not isinstance(results, list):
            results = [results]

        max_v = -np.inf
        min_v =  np.inf
        for line in results:
            v = np.array([state.v.mag() for state in line.raceline])

            max_v = max(max_v, v.max())
            min_v = min(min_v, v.min())

        linestyles = ['solid','dashed','dotted']
        for k,line in enumerate(results):
            sol_xi = np.array([state.x.xi for state in line.raceline])
            sol_xj = np.array([state.x.xj for state in line.raceline])
            sol_v  = np.array([state.v.mag() for state in line.raceline])

            points = np.array([sol_xi, sol_xj]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis = 1)

            norm = plt.Normalize(min_v, max_v)
            lc = LineCollection(segments, cmap='plasma', norm=norm, label = '%s (%0.2fs)' %(line.label, line.time), linestyles = linestyles[k%3])
            lc.set_array(sol_v)
            lc.set_linewidth(2)
            line = plt.gca().add_collection(lc)
        cbar = fig.colorbar(line, ax = plt.gca())
        cbar.set_label('Vehicle Speed (m/s)')
        plt.legend()

        plt.show(block = block)
        return

    def plot_raceline_3D(self, results, circular_buf = None, closed = None):
        RacelineWindow(self.surf, self.model, results, circular_buf = circular_buf, closed = closed)
        return

    def plot_raceline_two_track_details(self, results):
        s = np.array([state.p.s for state in results.raceline])
        Nfr = np.array([state.tfr.N for state in results.raceline])
        sfr = np.array([state.tfr.s for state in results.raceline])
        afr = np.array([state.tfr.a for state in results.raceline])
        Nfl = np.array([state.tfl.N for state in results.raceline])
        sfl = np.array([state.tfl.s for state in results.raceline])
        afl = np.array([state.tfl.a for state in results.raceline])
        Nrr = np.array([state.trr.N for state in results.raceline])
        srr = np.array([state.trr.s for state in results.raceline])
        arr = np.array([state.trr.a for state in results.raceline])
        Nrl = np.array([state.trl.N for state in results.raceline])
        srl = np.array([state.trl.s for state in results.raceline])
        arl = np.array([state.trl.a for state in results.raceline])
        colors = ['k','b','r','c']

        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(s,sfr, colors[0], label = 'Front Right')
        plt.plot(s,sfl, colors[1], label = 'Front Left')
        plt.plot(s,srr, colors[2], label = 'Rear Right')
        plt.plot(s,srl, colors[3], label = 'Rear Left')
        plt.ylabel('Slip Ratio')
        plt.gca().set_xticklabels([])
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(s,afr, colors[0])
        plt.plot(s,afl, colors[1])
        plt.plot(s,arr, colors[2])
        plt.plot(s,arl, colors[3])
        plt.ylabel('Slip Angle')
        plt.gca().set_xticklabels([])

        plt.subplot(3,1,3)
        plt.plot(s,Nfr, colors[0])
        plt.plot(s,Nfl, colors[1])
        plt.plot(s,Nrr, colors[2])
        plt.plot(s,Nrl, colors[3])
        plt.ylabel('Normal Force')
        plt.xlabel('Path Length')
        plt.show()



