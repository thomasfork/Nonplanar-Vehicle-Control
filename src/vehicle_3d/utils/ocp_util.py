'''
utilities for optimal control,
such as functions for solving a dynamic model over a particlar
space or time domain
'''
from abc import abstractmethod, ABC
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Callable, Union, Dict
import time
import warnings
import os
import sys

import numpy as np
import casadi as ca
import imgui

from vehicle_3d.pytypes import PythonMsg, BaseBodyState
from vehicle_3d.utils.discretization_utils import get_collocation_coefficients, \
    interpolate_collocation, interpolate_linear
from vehicle_3d.utils.ca_utils import ca_function_dispatcher, compile_solver, \
    compiler_available
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.dynamics_model import DynamicsModel, DAEDynamicsModel, \
    DynamicsModelConfig

from vehicle_3d.visualization.utils import get_instance_transforms, triangulate_trajectory
from vehicle_3d.visualization.objects import InstancedVertexObject, UBOObject, \
    VertexObject
from vehicle_3d.visualization.window import Window

class Method(Enum):
    ''' numerical method for enforcing dynamics model '''
    Euler = 0
    RK4 = 1
    COLLOCATION_LEGENDRE = 'legendre'
    COLLOCATION_RADAU    = 'radau'

COLLOCATION_METHODS = [Method.COLLOCATION_LEGENDRE, Method.COLLOCATION_RADAU]

class Fix(Enum):
    ''' what is to be fixed in optimization '''
    S = 0
    ''' fix s coordinate of all intervals '''
    T = 1
    ''' fix t coordinate of all intervals '''
    UNIFORM_DT = 2
    ''' uniform variable time for all intervals, ie. for point to point planning '''

@dataclass
class OCPConfig(PythonMsg):
    '''
    optimal control problem configuration

    For problems that move, ie. online control, set mobile to True. This will
    cause certainfixed inputs to be made parameters input to the solver.
    For instance for fixed space planning the surface parameters at each point
    become parameters and can thus change quickly

    '''
    method: Method = field(default = Method.COLLOCATION_LEGENDRE)
    fix: Fix = field(default = Fix.S)
    mobile: bool = field(default = False)
    compile: bool = field(default = False)
    verbose: bool = field(default = False)
    use_warmstart: bool = field(default = False)
    full_unpack: bool = field(default = True)
    # option to plot iterations, will slow down most problems
    plot_iterations: bool = field(default = False)
    max_iter: int = field(default = 1000)
    closed: bool = field(default = False) # if True, equate final and initial states and inputs

    N: int = field(default = 30)
    ''' number of intervals '''
    R: np.ndarray  = field(default = 1) # if a float - multiplied by identity matrix
    ''' input cost '''
    dR: np.ndarray = field(default = 1)
    ''' input rate cost '''

    # collocation-specific parameters
    K: int = field(default = 7)
    ''' order of collocation method '''
    tau: np.ndarray = field(default = None)
    B: np.ndarray = field(default = None)
    C: np.ndarray = field(default = None)
    D: np.ndarray = field(default = None)

    # fixed time parameters
    dt: float = field(default = 1)
    ''' time duration of intervals when time is fixed'''

    # initial guess parameters
    h_ws: float = field(default = 1)
    v_ws: float = field(default = 10)

@dataclass
class OCPResults(PythonMsg):
    ''' template for storing ocp results '''
    feasible: bool = field(default = None)
    periodic: bool = field(default = False)
    states: List[BaseBodyState] = field(default = None)
    t0: float = field(default = 0)
    step_sizes: np.ndarray = field(default = np.array([]))
    time: float = field(default = -1)
    label: str = field(default = '')
    color: np.ndarray = field(default = np.array([0,0,0,1]))
    z_interp: Callable[[float], np.ndarray] = field(default = None)
    u_interp: Callable[[float], np.ndarray] = field(default = None)
    du_interp: Callable[[float], np.ndarray] = field(default = None)
    a_interp: Callable[[float], np.ndarray] = field(default = None)
    solve_time: float = field(default = -1)
    ipopt_time: float = field(default = -1)
    feval_time: float = field(default = -1)

    def triangulate_trajectory(self,
            ubo: UBOObject,
            model: DynamicsModel,
            surf: BaseSurface,
            v_max: float = None,
            v_min: float = None,
            n: int = 1000) -> VertexObject:
        '''
        triangulate the trajectory, returning a 3D line along it, colored by speed
        v_min and v_max provide optional different colorbar limits
        n adjust the number of points in the trajectory
        '''
        V, I = self._VI_trajectory(model, surf, v_max, v_min, n)
        return VertexObject(ubo, V, I, simple=True)

    def update_triangulated_trajectory(self,
            obj: VertexObject,
            model: DynamicsModel,
            surf: BaseSurface,
            v_max: float = None,
            v_min: float = None,
            n: int = 1000) -> None:
        '''
        update a triangulated trajectory object that has already been created,
        such as if the trajectory has been recomputed
        '''
        obj.setup(*self._VI_trajectory(model, surf, v_max, v_min, n))

    def _VI_trajectory(self,
            model: DynamicsModel,
            surf: BaseSurface,
            v_max: float = None,
            v_min: float = None,
            n: int = 1000):
        t = np.linspace(self.t0, self.t0 + self.time, n)
        z = self.z_interp(t[None])
        u = self.u_interp(t[None])
        if not isinstance(model, DAEDynamicsModel):
            v = np.linalg.norm(model.f_vb(z, u), axis = 0)
            p = model.f_p(z,u)
        else:
            a = self.a_interp(t[None])
            v = np.linalg.norm(model.f_vb(z, u, a), axis = 0)
            p = model.f_p(z,u,a)

        x = surf.p2x_fast(p[0:1], p[1:2], p[2:3]).T
        e2 = surf.p2epp(p[0:1], p[1:2]).T
        e3 = surf.p2xpn(p[0:1], p[1:2]).T

        V, I, _, _ = triangulate_trajectory(
            x = x,
            e2 = e2,
            e3 = e3,
            v = v,
            v_max = v_max,
            v_min = v_min,
            closed = self.periodic,
            n = n
        )
        return V, I

    def triangulate_instanced_trajectory(self,
            ubo: UBOObject,
            t: np.ndarray,
            model: DynamicsModel,
            surf: BaseSurface) -> InstancedVertexObject:
        ''' update an instanced vertex object with instances at t '''
        R = self._trajectory_instances(t, model, surf)
        obj = model.get_instanced_visual_asset(ubo)
        obj.apply_instancing(R)
        return obj

    def update_instanced_trajectory(self,
            obj: InstancedVertexObject,
            t: np.ndarray,
            model: DynamicsModel,
            surf: BaseSurface):
        ''' update an instanced vertex object with instances at t'''
        R = self._trajectory_instances(t, model, surf)
        obj.apply_instancing(R)

    def _trajectory_instances(self,
            t: np.ndarray,
            model: DynamicsModel,
            surf: BaseSurface):
        z = self.z_interp(t[None])
        u = self.u_interp(t[None])
        if not isinstance(model, DAEDynamicsModel):
            r = model.f_R(z, u)
            p = model.f_p(z,u)
        else:
            a = self.a_interp(t[None])
            r = model.f_R(z,u,a)
            p = model.f_p(z,u,a)
        x = surf.p2x_fast(p[0:1], p[1:2], p[2:3]).T
        r = r.T.reshape((-1,3,3))
        r = r.transpose((0,2,1))
        R = get_instance_transforms(
            x,
            r = r)
        return R


@dataclass
class OCPVars(PythonMsg):
    ''' variables corresponding to ocp setup '''
    w: Union[ca.SX, ca.MX] = field(default = None)
    ''' full decision variable vector '''
    ubw: List[float] = field(default = None)
    lbw: List[float] = field(default = None)
    g: Union[ca.SX, ca.MX] = field(default = None)
    ''' full nonlinear constraint expression vector '''
    ubg: List[float] = field(default = None)
    lbg: List[float] = field(default = None)
    J: Union[ca.SX, ca.MX] = field(default = 0.)
    ''' total cost to minimize '''
    p: Union[ca.SX, ca.MX] = field(default = None)
    ''' optimization problem parameters, None if no parameters '''
    f_ode: ca.Function = field(default = None)
    state_dim: float = field(default = None)
    input_dim: float = field(default = None)
    param_dim: float = field(default = None)
    h_dae: ca.Function = field(default = None)
    alg_dim: float = field(default = None)
    dae_dim: float = field(default = None)

    H: np.ndarray = field(default = None)
    T: np.ndarray = field(default = None)
    Z: np.ndarray = field(default = None)
    U: np.ndarray = field(default = None)
    dU: np.ndarray = field(default = None)
    A: np.ndarray = field(default = None)
    P: np.ndarray = field(default = None)

    def __post_init__(self):
        self.w = []
        self.ubw = []
        self.lbw = []
        self.g = []
        self.ubg = []
        self.lbg = []

_ALLOWED_NONETYPE_VARS = ['p', 'P']
_ALLOWED_ODE_NONETYPE_VARS = ['h_dae', 'alg_dim', 'dae_dim', 'A']

class OCP(ABC):
    '''
    generic class for OCP
    helps set up, solve, and unpack problems
    primarily helps enforce a dynamics model over a problem domain
    '''
    surf: BaseSurface
    config: OCPConfig
    model: Union[DynamicsModel, DAEDynamicsModel]
    model_config: DynamicsModelConfig

    # optional fields for warmstart
    ws_results: OCPResults
    ws_model: Union[DynamicsModel, DAEDynamicsModel]

    dae_setup: bool
    sym_class: Union[ca.SX, ca.MX] = ca.SX
    ocp_vars: OCPVars

    setup_time: float = -1
    solve_time: float = -1
    ipopt_time: float = -1
    feval_time: float = -1

    solver: ca.nlpsol
    solver_w0: List[float]
    solver_ubw: List[float]
    solver_lbw: List[float]
    solver_ubg: List[float]
    solver_lbg: List[float]
    sol: Dict[str, ca.DM]

    z_interp: ca.Function
    u_interp: ca.Function
    du_interp: ca.Function
    a_interp: ca.Function

    f_sol_h: ca.Function
    f_sol_t: ca.Function
    f_sol_z: ca.Function
    f_sol_u: ca.Function
    f_sol_du: ca.Function

    # additional dae fields
    f_sol_a: ca.Function

    iteration_callback_fig = None

    # optional internal state, ie. for online control
    current_state: BaseBodyState = None

    def __init__(self,
            surf: BaseSurface,
            config: OCPConfig,
            model_config: DynamicsModelConfig,
            ws_results: OCPResults = None,
            ws_model: DynamicsModel = None,
            setup: bool = True):
        self.surf = surf
        self.config = config
        self.model_config = model_config
        self.ws_results = ws_results
        self.ws_model = ws_model
        self.model = self._get_model()
        self.dae_setup = isinstance(self.model, DAEDynamicsModel)
        if setup:
            self._setup()

    def step(self, state: BaseBodyState):
        '''
        intended method to step online controllers with a given current state
        '''
        self.current_state = state
        return self.solve()

    def solve(self):
        ''' solve the raceline problem, return results'''
        t0 = time.time()
        solver_opts = {'x0': self.solver_w0,
                       'ubx': self.solver_ubw,
                       'lbx': self.solver_lbw,
                       'ubg': self.solver_ubg,
                       'lbg': self.solver_lbg}
        if self.ocp_vars.p is not None:
            solver_opts['p'] = self._get_params()
        sol = self.solver(**solver_opts)
        self.solve_time = time.time() - t0
        self.sol = sol

        if self.config.verbose:
            sol_h = self.f_sol_h(sol['x'])
            print(f'reached target in    {np.sum(sol_h):0.3f} seconds')
            print(f'with a total cost of {float(sol["f"]):0.3f}')
            print(f'it took              {self.setup_time:0.3f} seconds to set up the problem')
            print(f'    and              {self.solve_time:0.3f} seconds to solve it')

        if self.config.plot_iterations:
            print('Warning - Timing statistics are incorrect when plotting iterations')
            while not self.iteration_callback_fig.should_close:
                self.iteration_callback_fig.draw()
            self.iteration_callback_fig.close()

        stats = self.solver.stats()
        self.feval_time = \
            stats['t_wall_nlp_f'] + \
            stats['t_wall_nlp_g'] + \
            stats['t_wall_nlp_grad_f'] + \
            stats['t_wall_nlp_hess_l'] + \
            stats['t_wall_nlp_jac_g']
        self.ipopt_time = self.solve_time - self.feval_time

        return self._unpack_soln(sol, solved=True)

    def get_ws(self):
        ''' get current initial guess, rather than solving'''
        self.solve_time = -1
        self.ipopt_time = -1
        self.feval_time = -1
        return self._unpack_soln({'x':self.solver_w0}, solved=False)

    @abstractmethod
    def _get_model(self) -> DynamicsModel:
        ''' get a dynamics model a given surface '''
        return None

    def _get_prediction_label(self) -> str:
        ''' get a string label for the prediction results'''
        return self.model.get_label()

    def _get_prediction_color(self) -> List[float]:
        ''' get a RGBA color for the prediction results, fields are between 0 and 1'''
        return self.model.get_color()

    def _ws_available(self) -> bool:
        return self.ws_model is not None and self.ws_results is not None

    def _pre_setup_checks(self):
        '''
        any checks that should be done before setting up the problem
        inheriting classes should add to the checks performed.
        '''
        if self.config.method in [Method.Euler, Method.RK4]:
            self.config.K = 0

        if self.config.fix == Fix.S and not self.config.method in COLLOCATION_METHODS:
            raise NotImplementedError('Cannot fix intervals in space for non-collocation method')

        if self.config.compile:
            if not compiler_available():
                warnings.warn('Valid compiler is not available, install GCC or MSVC')
                self.sym_class = ca.SX
                self.config.compile = False
            else:
                self.sym_class = ca.MX

        #NOTE - problems that use xp(s,y) must check the following:
        #if self.surf.config.mx_xp:
        #    self.sym_class = ca.MX

    def _post_setup_checks(self):
        ''' checks to run after nlp setup but before creating a solver '''
        for label, attr in asdict(self.ocp_vars).items():
            if attr is None and label not in _ALLOWED_NONETYPE_VARS:
                if not self.dae_setup and label in _ALLOWED_ODE_NONETYPE_VARS:
                    continue
                raise RuntimeError(f'Incomplete model setup, attribute {label} is NoneType')

    def _setup(self):
        t0 = time.time()
        self._pre_setup_checks()
        self._create_nlp()
        self._post_setup_checks()
        self._create_solver()
        self.setup_time = time.time() - t0

    def _create_nlp(self):
        self._initial_nlp_setup()
        self._assign_nlp_model()
        self._create_nlp_vars()
        self._enforce_model()
        self._add_costs()
        self._create_problem()

    def _initial_nlp_setup(self):
        self.ocp_vars = OCPVars()

    def _assign_nlp_model(self):
        ''' get a model for the dynamical system '''
        h_dae = 0
        if self.config.fix == Fix.S or self.config.mobile:
            f_ode = self.model.f_zdot_full
            if self.dae_setup:
                h_dae = self.model.f_h_full
            self.ocp_vars.param_dim = self.surf.sym_rep.param_dim
        else:
            f_ode = self.model.f_zdot
            if self.dae_setup:
                h_dae = self.model.f_h
            self.ocp_vars.param_dim = 0

        self.ocp_vars.f_ode = f_ode
        self.ocp_vars.state_dim = self.model.model_vars.state_dim
        self.ocp_vars.input_dim = self.model.model_vars.input_dim

        if self.dae_setup:
            self.ocp_vars.h_dae = h_dae
            self.ocp_vars.alg_dim = self.model.model_vars.alg_dim
            self.ocp_vars.dae_dim = self.model.model_vars.dae_dim
        self._nlp_model_setup_checks()

    def _nlp_model_setup_checks(self):
        input_dim = self.ocp_vars.input_dim
        # if cost matrices have been provided as a scalar, convert to a matrix
        # done here since previously the correct dimensions are unknown
        if isinstance(self.config.R, (float, int)):
            self.config.R = np.eye(input_dim) * self.config.R

        if isinstance(self.config.dR, (float, int)):
            self.config.dR = np.eye(input_dim) * self.config.dR

    def _create_nlp_vars(self):
        N = self.config.N
        K = self.config.K
        if self.config.method in COLLOCATION_METHODS:
            self.config.tau, self.config.B, self.config.C, self.config.D = \
                get_collocation_coefficients(K, method = self.config.method.value)

        var_shape = (N,K+1)
        H  = np.resize(np.array([], dtype = self.sym_class), (N))        # interval step size
        T  = np.resize(np.array([], dtype = self.sym_class), var_shape)  # time from start
        Z  = np.resize(np.array([], dtype = self.sym_class), var_shape)  # differential state
        U  = np.resize(np.array([], dtype = self.sym_class), var_shape)  # input
        dU = np.resize(np.array([], dtype = self.sym_class), var_shape)  # input rate
        A = np.resize(np.array([], dtype = self.sym_class), var_shape)   # algebraic state
        P = np.resize(np.array([], dtype = self.sym_class), var_shape)   # parameters

        # step size variables
        if self.config.fix == Fix.S:
            for n in range(N):
                hk = self.sym_class.sym(f'h_{n}')
                H[n] = hk
        elif self.config.fix == Fix.UNIFORM_DT:
            h = self.sym_class.sym('h')
            for n in range(N):
                H[n] = h
        elif self.config.fix == Fix.T:
            for n in range(N):
                H[n] = self.config.dt
        else:
            raise NotImplementedError(f'Unrecognized fix type {self.config.fix}')

        # other variables
        T[0,0] = 0
        for n in range(N):
            if n > 0:
                T[n,0] = T[n-1, 0] + H[n-1]
            for k in range(K+1):
                if k > 0:
                    T[n,k] = self.config.tau[k] * H[n] + T[n,0]
                Z[n,k]  = self.sym_class.sym(f'z_{n}_{k}',  self.ocp_vars.state_dim)
                U[n,k]  = self.sym_class.sym(f'u_{n}_{k}',  self.ocp_vars.input_dim)
                dU[n,k] = self.sym_class.sym(f'du_{n}_{k}', self.ocp_vars.input_dim)
                if self.dae_setup:
                    A[n,k] = self.sym_class.sym(f'a_{n}_{k}', self.ocp_vars.alg_dim)
                if self.config.mobile:
                    P[n,k] = self.sym_class.sym(f'p_{n}_{k}', self.ocp_vars.param_dim)

        self.ocp_vars.H = H
        self.ocp_vars.T = T
        self.ocp_vars.Z = Z
        self.ocp_vars.U = U
        self.ocp_vars.dU = dU
        if self.dae_setup:
            self.ocp_vars.A = A
        if self.config.mobile:
            self.ocp_vars.P = P

    def _get_s(self, n:int, k:int):
        if not self.config.fix == Fix.S:
            return 0
        s0 = self.surf.s_min()
        sf = self.surf.s_max()
        ds = (sf - s0) / self.config.N

        return s0 + ds * (n + self.config.tau[k])

    def _eval_ode(self, n:int, k:int = 0):
        ''' evaluate ODE on decision variables at n,k; add DAE constraints if pertinent '''
        Z = self.ocp_vars.Z
        U = self.ocp_vars.U
        A = self.ocp_vars.A

        dae_eval = 0

        generic_args = [Z[n,k], U[n,k]]
        if self.dae_setup:
            generic_args += [A[n,k]]

        ode_eval, dae_eval = self._ode_helper(n,k,generic_args)

        if self.dae_setup:
            self.ocp_vars.g += [dae_eval]
            self.ocp_vars.ubg += [0.] * self.ocp_vars.dae_dim
            self.ocp_vars.lbg += [0.] * self.ocp_vars.dae_dim

        return ode_eval

    def _ode_helper(self, n:int,k:int, ode_args):
        f_ode = self.ocp_vars.f_ode
        h_dae = self.ocp_vars.h_dae
        dae_eval = None
        if self.config.mobile:
            param_terms = self.ocp_vars.P[n,k]
            ode_eval = f_ode(*ode_args, param_terms)
            if self.dae_setup:
                dae_eval = h_dae(*ode_args, param_terms)

        elif self.config.fix == Fix.S:
            s = self._get_s(n, k)
            param_terms = self.surf.sym_rep.f_param_terms(s,0,0)
            ode_eval = f_ode(*ode_args, param_terms)
            if self.dae_setup:
                dae_eval = h_dae(*ode_args, param_terms)

        else:
            ode_eval = f_ode(*ode_args)
            if self.dae_setup:
                dae_eval = h_dae(*ode_args)
        return ode_eval, dae_eval

    def _enforce_model(self):
        for n in range(self.config.N):
            if self.config.method in COLLOCATION_METHODS:
                self._enforce_collocation_interval(n)
            elif self.config.method == Method.Euler:
                if n < self.config.N -1:
                    self._enforce_euler(n)
            elif self.config.method == Method.RK4:
                if n < self.config.N -1:
                    self._enforce_rk4(n)
            else:
                raise NotImplementedError('unrecognized ode method')
            for k in range(self.config.K + 1):
                self._enforce_stage_constraints(n, k)

        if self.config.closed:
            self._enforce_loop_closure()
        else:
            self._enforce_initial_constraints()
            self._enforce_terminal_constraints()

    def _zF(self) -> ca.SX:
        ''' last state of last interval '''
        if self.config.method in COLLOCATION_METHODS:
            K = self.config.K
            Z = self.ocp_vars.Z
            D = self.config.D
            zF = 0
            for k in range(K+1):
                zF += Z[-1,k] * D[k]
            return zF
        return self.ocp_vars.Z[-1, 0]

    def _uF(self) -> ca.SX:
        ''' last input of last interval '''
        if self.config.method in COLLOCATION_METHODS:
            K = self.config.K
            U = self.ocp_vars.U
            D = self.config.D
            uF = 0
            for k in range(K+1):
                uF += U[-1,k] * D[k]
            return uF
        return self.ocp_vars.U[-1, 0]

    def _aF(self) -> ca.SX:
        ''' last algebraic state of last interval '''
        if not self.dae_setup:
            return None
        if self.config.method in COLLOCATION_METHODS:
            K = self.config.K
            A = self.ocp_vars.A
            D = self.config.D
            aF = 0
            for k in range(K+1):
                aF += A[-1,k] * D[k]
            return aF
        return self.ocp_vars.A[-1, 0]

    def _state_continuity_operator(self, z: Union[ca.SX, ca.MX]):
        '''
        any additional operations to do on states between intervals
        ex: quaternion normalization
        '''
        return z

    def _enforce_euler(self, n:int):
        z_new = self._state_continuity_operator(
            self.ocp_vars.Z[n,0] + self._eval_ode(n,0) * self.ocp_vars.H[n]
        )
        self.ocp_vars.g += [self.ocp_vars.Z[n+1,0] - z_new]
        self.ocp_vars.ubg += [0.] * self.ocp_vars.state_dim
        self.ocp_vars.lbg += [0.] * self.ocp_vars.state_dim

    def _enforce_rk4(self, n:int):
        dt = self.ocp_vars.H[n]
        z0 = self.ocp_vars.Z[n,0]
        u = self.ocp_vars.U[n,0]
        ode_args = [z0, u]
        if self.dae_setup:
            ode_args += [self.ocp_vars.A[n,0]]

        k1 = self._eval_ode(n,0)
        ode_args[0] = z0 + dt/2*k1
        k2, _ = self._ode_helper(n,0, ode_args)
        ode_args[0] = z0 + dt/2*k2
        k3, _ = self._ode_helper(n,0, ode_args)
        ode_args[0] = z0 + dt*k3
        k4, _ = self._ode_helper(n,0, ode_args)

        z_new  = z0 + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

        self.ocp_vars.g += [self.ocp_vars.Z[n+1,0] - z_new]
        self.ocp_vars.ubg += [0.] * self.ocp_vars.state_dim
        self.ocp_vars.lbg += [0.] * self.ocp_vars.state_dim

    def _enforce_collocation_interval(self, n:int):
        self._enforce_collocation_interval_ode(n)
        self._enforce_collocation_interval_continuity(n)

    def _enforce_collocation_interval_ode(self, n:int):
        ''' enforce ODE throughout interval '''
        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        for k in range(self.config.K+1):
            poly_ode = 0
            poly_du = 0
            for k2 in range(self.config.K+1):
                poly_ode += self.config.C[k2][k] * self.ocp_vars.Z[n,k2] / self.ocp_vars.H[n]
                poly_du  += self.config.C[k2][k] * self.ocp_vars.U[n,k2] / self.ocp_vars.H[n]

            func_ode = self._eval_ode(n,k)

            if k > 0:
                g += [func_ode - poly_ode]
                ubg += [0.] * self.ocp_vars.state_dim
                lbg += [0.] * self.ocp_vars.state_dim

            g += [self.ocp_vars.dU[n,k] - poly_du]
            ubg += [0.] * self.ocp_vars.input_dim
            lbg += [0.] * self.ocp_vars.input_dim

    def _enforce_collocation_interval_continuity(self, n:int):
        ''' enforce continuity with the previous interval '''
        if n == 0:
            return
        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        poly_prev_state = 0
        poly_prev_input = 0
        for k in range(self.config.K+1):
            poly_prev_state += self.ocp_vars.Z[n-1,k] * self.config.D[k]
            poly_prev_input += self.ocp_vars.U[n-1,k] * self.config.D[k]

        poly_prev_state = self._state_continuity_operator(poly_prev_state)
        if self.config.fix == Fix.S:
            g += [self.ocp_vars.Z[n,0][1:] - poly_prev_state[1:]]
            ubg += [0.] * (self.ocp_vars.state_dim-1)
            lbg += [0.] * (self.ocp_vars.state_dim-1)
        else:
            g += [self.ocp_vars.Z[n,0] - poly_prev_state]
            ubg += [0.] * self.ocp_vars.state_dim
            lbg += [0.] * self.ocp_vars.state_dim

        g += [self.ocp_vars.U[n,0] - poly_prev_input]
        ubg += [0.] * self.ocp_vars.input_dim
        lbg += [0.] * self.ocp_vars.input_dim

    def _enforce_loop_closure(self):
        ''' loop closure constraints on state and input'''
        g = self.ocp_vars.g
        ubg = self.ocp_vars.ubg
        lbg = self.ocp_vars.lbg

        zF = self._state_continuity_operator(self._zF())

        g += [self._uF() - self.ocp_vars.U[0,0]]
        ubg += [0.] * self.ocp_vars.input_dim
        lbg += [0.] * self.ocp_vars.input_dim
        if self.config.fix == Fix.S:
            g += [zF[1:] - self.ocp_vars.Z[0,0][1:]]
            ubg += [0.] * (self.ocp_vars.state_dim-1)
            lbg += [0.] * (self.ocp_vars.state_dim-1)
        else:
            g += [zF - self.ocp_vars.Z[0,0]]
            ubg += [0.] * self.ocp_vars.state_dim
            lbg += [0.] * self.ocp_vars.state_dim

    def _enforce_initial_constraints(self):
        ''' initial constraints '''

    def _enforce_terminal_constraints(self):
        ''' terminal constraints '''

    def _enforce_stage_constraints(self, n:int, k:int):
        ''' constraints to enforce at individual stages '''
        inputs = [self.ocp_vars.Z[n,k], self.ocp_vars.U[n,k]]
        if self.dae_setup:
            inputs += [self.ocp_vars.A[n,k]]

        if self.config.mobile:
            param_terms = self.ocp_vars.P[n,k]
        elif self.config.fix == Fix.S:
            s = self._get_s(n, k)
            param_terms = self.surf.sym_rep.f_param_terms(s,0,0)
        else:
            param_terms = None

        self.model.add_model_stage_constraints(
            inputs,
            self.ocp_vars.g,
            self.ocp_vars.lbg,
            self.ocp_vars.ubg,
            param_terms = param_terms
        )

        # fixed s coordinate at the start and end of every interval
        if k == 0 and self.config.fix == Fix.S:
            self.ocp_vars.g += [self.ocp_vars.Z[n,k][0] - self._get_s(n,k)]
            self.ocp_vars.ubg += [0.]
            self.ocp_vars.lbg += [0.]

            if self.config.method in COLLOCATION_METHODS:
                end_state = 0
                for k2 in range(self.config.K+1):
                    end_state += self.ocp_vars.Z[n,k2] * self.config.D[k2]

                self.ocp_vars.g += [end_state[0] - self._get_s(n+1,0)]
                self.ocp_vars.ubg += [0.]
                self.ocp_vars.lbg += [0.]

    def _add_costs(self):
        for n in range(self.config.N):
            for k in range(self.config.K+1):
                self._add_stage_cost(n,k)
        self._add_terminal_cost()

    def _add_stage_cost(self, n:int, k:int):
        if self.config.method in COLLOCATION_METHODS:
            self.ocp_vars.J += self._stage_cost(n,k) * self.ocp_vars.H[n] * self.config.B[k]
        else:
            self.ocp_vars.J += self._stage_cost(n,k) * self.ocp_vars.H[n]

    def _stage_cost(self, n:int, k:int):
        ''' stage cost, default is time plus regularization '''
        return ca.bilin(self.config.R, self.ocp_vars.U[n,k], self.ocp_vars.U[n,k]) + \
               ca.bilin(self.config.dR, self.ocp_vars.dU[n,k], self.ocp_vars.dU[n,k]) + \
               1

    def _add_terminal_cost(self):
        ''' terminal cost '''

    def _create_problem(self):
        ''' finish building problem variables '''
        w, w0, ubw, lbw = self._build_decision_vector()
        w = ca.vertcat(*w)

        self.ocp_vars.g = ca.vertcat(*self.ocp_vars.g)
        self.ocp_vars.w = w
        self.ocp_vars.ubw = ubw
        self.ocp_vars.lbw = lbw

        self._add_params()

        self.solver_w0 = w0
        self.solver_ubw = ubw
        self.solver_lbw = lbw
        self.solver_ubg = self.ocp_vars.ubg
        self.solver_lbg = self.ocp_vars.lbg

        if self.config.method in COLLOCATION_METHODS:
            self.z_interp = interpolate_collocation(w, self.ocp_vars.H, self.ocp_vars.Z,
                                                    self.config)
            self.u_interp = interpolate_collocation(w, self.ocp_vars.H, self.ocp_vars.U,
                                                    self.config)
            self.du_interp =interpolate_collocation(w, self.ocp_vars.H, self.ocp_vars.dU,
                                                    self.config)
        else:
            self.z_interp = interpolate_linear(w, self.ocp_vars.H, self.ocp_vars.Z)
            self.u_interp = interpolate_linear(w, self.ocp_vars.H, self.ocp_vars.U)
            self.du_interp = interpolate_linear(w, self.ocp_vars.H, self.ocp_vars.dU)

        if self.dae_setup:
            self.a_interp = interpolate_linear(w, self.ocp_vars.H, self.ocp_vars.A)

        def packer(X):
            '''
            function to pack problem data into CasADi format for unpacking
            '''
            return ca.horzcat(*[ca.horzcat(*X[k]) for k in range(X.shape[0])]).T

        self.f_sol_h = ca_function_dispatcher(ca.Function('h', [w], self.ocp_vars.H.tolist()))
        self.f_sol_t = ca_function_dispatcher(ca.Function('t', [w], [packer(self.ocp_vars.T)]))
        self.f_sol_z = ca_function_dispatcher(ca.Function('z', [w], [packer(self.ocp_vars.Z)]))
        self.f_sol_u = ca_function_dispatcher(ca.Function('u', [w], [packer(self.ocp_vars.U)]))
        self.f_sol_du = ca_function_dispatcher(ca.Function('du', [w], [packer(self.ocp_vars.dU)]))
        if self.dae_setup:
            self.f_sol_a = ca_function_dispatcher(ca.Function('t', [w], [packer(self.ocp_vars.A)]))

    def _build_decision_vector(self):
        N = self.config.N
        K = self.config.K
        H = self.ocp_vars.H

        w = []
        w0 = []
        ubw = []
        lbw = []

        # add any time variables
        if self.config.fix == Fix.S:
            for n in range(N):
                w += [H[n]]
                h0 = self._guess_h(n)
                ubw += [h0*40]
                lbw += [h0/40]
                w0 += [h0]
        elif self.config.fix == Fix.UNIFORM_DT:
            w += [H[0]]
            h0 = self.config.h_ws
            ubw += [h0*40]
            lbw += [h0/40]
            w0 += [h0]

        for n in range(N):
            for k in range(0, K+1):
                self._add_stage_decision_variables(n, k, w, w0, ubw, lbw)

        return w, w0, ubw, lbw

    def _add_stage_decision_variables(self, n:int, k:int, w, w0, ubw, lbw):
        state_u, state_l = self._get_state_bounds(n,k)
        input_u, input_l = self._get_input_bounds(n,k)
        input_du, input_dl = self._get_input_rate_bounds(n,k)

        w += [self.ocp_vars.Z[n,k]]
        lbw += state_l
        ubw += state_u

        w += [self.ocp_vars.U[n,k]]
        lbw += input_l
        ubw += input_u

        w += [self.ocp_vars.dU[n,k]]
        lbw += input_dl
        ubw += input_du

        w0 += self._guess_z(n,k)
        w0 += self._guess_u(n,k)
        w0 += self._guess_du(n,k)

        if self.dae_setup:
            w += [self.ocp_vars.A[n,k]]
            alg_u, alg_l = self._get_alg_state_bounds(n,k)
            ubw += alg_u
            lbw += alg_l
            w0 += self._guess_a(n,k)

    def _guess_h(self, n:int):
        # pylint: disable=unused-argument
        return self.config.h_ws

    def _guess_z(self, n:int, k:int):
        z = [0.] * self.ocp_vars.state_dim
        if self.config.fix == Fix.S:
            z[0] = self._get_s(n,k)
        return z

    def _guess_u(self, n:int, k:int):
        # pylint: disable=unused-argument
        return [0.] * self.ocp_vars.input_dim

    def _guess_du(self, n:int, k:int):
        # pylint: disable=unused-argument
        return [0.] * self.ocp_vars.input_dim

    def _guess_a(self, n:int, k:int):
        # pylint: disable=unused-argument
        return [0.] * self.ocp_vars.alg_dim

    def _get_state_bounds(self, n:int,k:int):
        return self.model.zu(self._get_s(n,k)), self.model.zl(self._get_s(n,k))

    def _get_input_bounds(self, n:int,k:int):
        # pylint: disable=unused-argument
        return self.model.uu(), self.model.ul()

    def _get_input_rate_bounds(self, n:int,k:int):
        # pylint: disable=unused-argument
        return self.model.duu(), self.model.dul()

    def _get_alg_state_bounds(self, n:int, k:int):
        # pylint: disable=unused-argument
        return self.model.au(), self.model.al()

    def _add_params(self):
        ''' add any (potentially none) parameters to the problem '''
        if self.config.mobile:
            self.ocp_vars.p = ca.vertcat(*np.concatenate(self.ocp_vars.P))

    def _get_params(self):
        ''' get parameters for a parameterized solver '''
        raise NotImplementedError('Must be implemented by child class')

    def _create_solver(self):
        prob = {}
        prob['x'] = self.ocp_vars.w
        prob['g'] = self.ocp_vars.g
        prob['f'] = self.ocp_vars.J
        if self.ocp_vars.p is not None:
            prob['p'] = self.ocp_vars.p

        if self.config.verbose:
            opts = {'ipopt.sb':'yes'}
        else:
            opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}

        if os.path.exists('/usr/local/lib'):
            if 'libcoinhsl.so' in os.listdir('/usr/local/lib/') and \
                    '3.6' in ca.__version__:
                # hsllib option is only supported on newer ipopt versions
                opts['ipopt.linear_solver'] = 'ma97'
                opts['ipopt.hsllib'] = '/usr/local/lib/libcoinhsl.so'
            elif 'libhsl.so' in os.listdir('/usr/local/lib/'):
                # check for obsolete hsl install and that it is on search path
                if '/usr/local/lib' in os.environ['LD_LIBRARY_PATH']:
                    opts['ipopt.linear_solver'] = 'ma97'
                else:
                    print('ERROR - HSL is present but not on path')
                    print('Run "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/"')
                    print('Defaulting to MUMPS')

        elif '3.6' in ca.__version__:
            print('WARNING - Casadi 3.6.X default linear solver for IPOPT is unreliable')
            print('Downgrading or using HSL solvers will be necessary for advanced racelines')

        opts['ipopt.max_iter'] = self.config.max_iter
        if '3.6' in ca.__version__:
            opts['ipopt.timing_statistics'] = 'yes'
        opts['ipopt.honor_original_bounds'] = 'yes'

        if self.config.plot_iterations:
            callback = IterationCallbackWindow(self.surf, prob, self._unpack_soln, self)
            self.iteration_callback_fig = callback
            opts['iteration_callback'] = callback

        if self.config.compile:
            # option-free solver so option changes don't cause new codegen
            self.solver = compile_solver('solver', 'ipopt', prob, opts)
        else:
            self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    def _unpack_soln(self, sol, solved:bool = False):
        '''
        unpacks the planned solution:
            state at each interval (including normal forces, global pose, etc..)
            interpolation objects for state, input, input rate, and dae state (if applicable)

        for fast online control, unpacking the state may be undesirable,
        as this can take ~ 1ms per state even for simple models
        this is toggled by self.config.full_unpack

        setting solved=False is intended for solve callback figures and unpacking warmstart
        information
        '''
        s0 = 0
        t0 = 0
        if self.current_state is not None:
            t0 = self.current_state.t
            # add path length offset to correctly interpolate mobile problems
            if self.config.mobile and self.config.fix == Fix.S:
                s0 = self.current_state.p.s

        sol_h = self.f_sol_h(sol['x'])

        states = []
        if self.config.full_unpack:
            sol_t = self.f_sol_t(sol['x']) + t0
            sol_z = self.f_sol_z(sol['x'])
            sol_z[:,0] += s0
            sol_u = self.f_sol_u(sol['x'])

            zip_args = [sol_t, sol_z, sol_u]
            if self.dae_setup:
                sol_a = self.f_sol_a(sol['x'])
                zip_args += [sol_a]

            for t, *args in zip(*zip_args):
                state = self.model.get_empty_state()
                state.t = t
                if self.dae_setup:
                    self.model.zua2state(state, *args)
                else:
                    self.model.zu2state(state, *args)
                states.append(state)

        t = self.sym_class.sym('t')

        z_interp = self.z_interp.call([t - t0, sol['x']])
        z_interp[0][0] += s0
        z_interp =  ca_function_dispatcher(ca.Function('z_interp', [t], z_interp))

        u_interp = self.u_interp.call([t - t0, sol['x']])
        u_interp = ca_function_dispatcher(ca.Function('u_interp', [t], u_interp))

        du_interp = self.du_interp.call([t - t0, sol['x']])
        du_interp = ca_function_dispatcher(ca.Function('du_interp', [t], du_interp))

        a_interp = None
        if self.dae_setup:
            a_interp = self.a_interp.call([t - t0, sol['x']])
            a_interp = ca_function_dispatcher(ca.Function('a_interp', [t], a_interp))

        feasible = False
        if solved:
            feasible = self.solver.stats()['success']

        return OCPResults(
            states = states,
            step_sizes = sol_h,
            time = np.sum(sol_h),
            t0 = t0,
            label = self._get_prediction_label(),
            color = self._get_prediction_color(),
            z_interp = z_interp,
            u_interp = u_interp,
            du_interp = du_interp,
            a_interp = a_interp,
            solve_time = self.solve_time,
            ipopt_time = self.ipopt_time,
            feval_time = self.feval_time,
            feasible = feasible)


class IterationCallbackWindow(ca.Callback, Window):
    ''' window for plotting intermediate solutions '''
    # pylint: disable=arguments-differ
    trajectory: VertexObject = None
    vehicles: InstancedVertexObject = None
    running: bool = False
    auto_advance_solver: bool = True
    steps_to_advance: int = 1
    steps_advanced: int = 0
    iteration: int = 0

    def __init__(self, surf: BaseSurface, prob: Dict,
                 unpacker: Callable[[ca.DM, bool], OCPResults],
                 solver: OCP):
        ca.Callback.__init__(self)
        self.surf = surf
        self.unpacker = unpacker
        self.solver = solver

        self.nx = prob['x'].shape[0]
        self.ng = prob['g'].shape[0]
        self.np = 0

        # Initialize internal objects
        self.construct('ocp_callback_fig', {})

        Window.__init__(self, surf)

        self._update_trajectory(unpacker({'x':solver.solver_w0}, False))
        while not self.running:
            self.draw()
            if self.should_close:
                self.close()
                return

    def get_n_in(self):
        ''' required by casadi '''
        return ca.nlpsol_n_out()

    def get_n_out(self):
        ''' required by casadi '''
        return 1

    def get_name_in(self, i):
        ''' required by casadi '''
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        # pylint: disable=unused-argument
        ''' required by casadi '''
        return "ret"

    def get_sparsity_in(self, i):
        ''' required by casadi '''
        n = ca.nlpsol_out(i)
        if n=='f':
            return ca.Sparsity. scalar()
        elif n in ('x', 'lam_x'):
            return ca.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity(0,0)

    def eval(self, arg):
        ''' called by casadi at each iteration'''
        # check that window is open and should not be closed
        if not self.window:
            return [0]
        if self.should_close:
            self.close()

        # Create dictionary
        darg = {}
        for (i,s) in enumerate(ca.nlpsol_out()):
            darg[s] = arg[i]

        self.steps_advanced += 1
        self.iteration += 1
        self._update_trajectory(self.unpacker(darg, False))
        return [0]

    def _update_trajectory(self, trajectory: OCPResults):
        if self.trajectory is None:
            self.trajectory = trajectory.triangulate_trajectory(
                self.ubo, self.solver.model, self.surf)
            self.add_object('trajectory', self.trajectory)
        else:
            trajectory.update_triangulated_trajectory(self.trajectory, self.solver.model, self.surf)

        t = np.linspace(0, trajectory.time, 50)
        if self.vehicles is None:
            self.vehicles =  trajectory.triangulate_instanced_trajectory(
                self.ubo,
                t,
                self.solver.model,
                self.surf
            )
            self.add_object('vehicles', self.vehicles)
        else:
            trajectory.update_instanced_trajectory(
                self.vehicles, t, self.solver.model, self.surf
            )

        self.draw()
        if not (self.auto_advance_solver and self.steps_advanced >= self.steps_to_advance)\
                or not self.running:
            self.running = False
            while not self.running:
                self.draw()
                if self.should_close:
                    self.close()
                    return

    def draw_extras(self):
        imgui.set_next_window_position(0,0)
        imgui.set_next_window_size(300, 0)

        imgui.begin("Solver Console",
            closable = False,
            flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR \
                | imgui.WINDOW_NO_COLLAPSE)

        imgui.text(f'Solver Iteration :{self.iteration}')

        run_btn_label = 'Run Solver' if self.auto_advance_solver else 'Advance Solver'
        if imgui.radio_button(run_btn_label, self.running):
            self.running = not self.running
            if self.running:
                self.steps_advanced = 0

        if imgui.radio_button('Auto Advance Solver', self.auto_advance_solver):
            self.auto_advance_solver = not self.auto_advance_solver

        if not self.auto_advance_solver:
            _, self.steps_to_advance = imgui.input_int('Steps', self.steps_to_advance)

        if imgui.button('Exit without solving'):
            sys.exit()

        imgui.end()
