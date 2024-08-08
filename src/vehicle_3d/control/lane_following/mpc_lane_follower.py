''' a simple mpc lane follower '''
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import casadi as ca

from vehicle_3d.pytypes import PythonMsg
from vehicle_3d.utils.ca_utils import compile_solver
from vehicle_3d.models.tangent_vehicle_model import KinematicVehicleModel, \
    TangentVehicleState
from vehicle_3d.control.base_controller import Controller


@dataclass
class NonplanarMPCConfig(PythonMsg):
    ''' MPC config '''
    N:  int      = field(default = 20)
    Q:  np.array = field(default = 100*np.diag([0,1,1,1]))
    R:  np.array = field(default = 0.0 * np.eye(3))
    dR: np.array = field(default = .1* np.eye(3))
    P:  np.array = field(default = 100*np.diag([0,1,1,1]))

    # simulation time step and time between sucessive calls of MPC
    dt:  float = field(default = 0.05)
    # time step for MPC prediction, not the frequency of the controller
    dtp: float = field(default = 0.05)

    ua_min:   float = field(default =  0)
    ua_max:   float = field(default =  10)
    dua_min:  float = field(default = -np.inf)
    dua_max:  float = field(default =  np.inf)
    ub_min:   float = field(default =  0)
    ub_max:   float = field(default =  10)
    dub_min:  float = field(default = -np.inf)
    dub_max:  float = field(default =  np.inf)
    uy_min:   float = field(default = -0.5)
    uy_max:   float = field(default =  0.5)
    duy_min:  float = field(default = -np.inf)
    duy_max:  float = field(default =  np.inf)
    s_min:   float = field(default = -np.inf)
    s_max:   float = field(default =  np.inf)
    y_min:   float = field(default = -4)
    y_max:   float = field(default =  4)
    V_min:   float = field(default = -20)
    V_max:   float = field(default =  20)
    ths_min: float = field(default = -2)
    ths_max: float = field(default =  2)

    yref: float   = field(default = 0) # lateral offset from centerline
    vref: float   = field(default = 10) # target speed

    use_planar_setup: bool = field(default = False) # set up planar control instead of nonplanar

    Nmin: float   = field(default = 8000) # target minimum normal force
    Nmax: float   = field(default = 40000) # target minimum normal force
    planner_ds: float = field(default = .5)    # spacing of planner points
    planner_N:  int   = field(default = 60)   # number  of planner points

    #generate and compile code using CasADi,
    # if false will only recompile if binaries not found.
    # since costs, reference speeds, and the surface are
    # all currently intrinsic to the solver this has to be done frequently.
    recompile:  bool = field(default = False)

class NonplanarMPC(Controller):
    ''' example MPC implementation '''
    planner_x0: ca.DM
    solver_x0:  ca.DM
    predicted_du: np.ndarray
    predicted_u:  np.ndarray
    predicted_x:  np.ndarray

    def __init__(self, model: KinematicVehicleModel, config: Optional[NonplanarMPCConfig] = None):
        if config is None:
            config = NonplanarMPCConfig()

        self.config = config
        self.model = model
        self.build_mpc_delta()
        if not self.config.use_planar_setup:
            self.build_mpc_planner()
            self.vref = self.config.vref

    def step(self, state:TangentVehicleState):
        z0, u0 = self.model.state2zu(state)
        zref =  [0, self.config.yref, 0, self.config.vref]
        duref = [0,0,0]

        t0 = time.time()
        if not self.config.use_planar_setup:
            plan = self.planner(x0  = self.planner_x0,
                                lbx = self.planner_lbx,
                                ubx = self.planner_ubx,
                                lbg = self.planner_lbg,
                                ubg = self.planner_ubg,
                                p = z0+u0 + [self.vref])
            self.planner_x0 = plan['x']
            zref[3] = float(plan['x'][0])
            self.vref = zref[3]


        sol = self.solver(x0  = self.solver_x0,
                          lbx = self.solver_lbx,
                          ubx = self.solver_ubx,
                          lbg = self.solver_lbg,
                          ubg = self.solver_ubg,
                          p = z0+u0+zref+duref)

        tf = time.time()

        t_sol = tf - t0

        state.u.a = float(sol['x'][3])
        state.u.b = float(sol['x'][4])
        state.u.y = float(sol['x'][5])

        if self.config.use_planar_setup:
            state.u.a += float(state.q.e1()[2]) * 9.81

        self.solver_x0 = sol['x']

        return t_sol

    def build_mpc_delta(self):
        '''
        sets up an mpc problem but with delta formulation in the input
        this helps with going up steep hills, where an affine input is necessary
        and in general where we'd rather penalize input rate over input magnitude
        in effect this tries to "automatically" attain an equillibrium input.

        input magnitude can still be penalized by config.R,
        and delta_input is penalized with config.dR
        it is recommended that R << dR
        '''

        z_size = self.model.model_vars.state_dim
        u_size = self.model.model_vars.input_dim

        f_zdot = self.model.f_zdot
        z0 = ca.MX.sym('z0',z_size)
        u0 = ca.MX.sym('u0',u_size)
        k1 = f_zdot(z0, u0)
        k2 = f_zdot(z0 + self.config.dtp/2*k1, u0)
        k3 = f_zdot(z0 + self.config.dtp/2*k2, u0)
        k4 = f_zdot(z0 + self.config.dtp*  k3, u0)
        zn  = z0  + self.config.dtp / 6 * (k1 + 2*k2 + 2*k3 + k4)
        self.F = ca.Function('F', [z0,u0], [zn], ['z0','u'],['zf'])


        p = []       # parameters (initial state, cost terms, terminal terms)
        w = []       # states     (vehicle state and inputs)
        w0 = []
        lbw = []
        ubw = []
        J = 0        # cost
        g = []       # nonlinear constraint functions
        lbg = []
        ubg = []

        zl = [self.config.s_min, self.config.y_min, self.config.ths_min, self.config.V_min]
        zu = [self.config.s_max, self.config.y_max, self.config.ths_max, self.config.V_max]
        zg  = [0] * z_size

        ul = [self.config.ua_min, self.config.ub_min, self.config.uy_min]
        uu = [self.config.ua_max, self.config.ub_max, self.config.uy_max]
        ug  = [0] * u_size

        dul = [self.config.dua_min, self.config.dub_min, self.config.duy_min]
        duu = [self.config.dua_max, self.config.dub_max, self.config.duy_max]
        dug = [0] * u_size

        z = ca.MX.sym('z0',z_size)
        u = ca.MX.sym('up',u_size)
        zref  = ca.MX.sym('zref',z_size)
        duref = ca.MX.sym('duref',u_size)

        p += [z]
        p += [u]
        p += [zref]
        p += [duref]

        for k in range(self.config.N):
            # add delta state
            du = ca.MX.sym('du' + str(k), u_size)
            w += [du]
            lbw += dul
            ubw += duu
            w0  += dug
            # separate dt for change from previous input compared to future prediction spacing
            if k == 0:
                unew = u + du * self.config.dt
            else:
                unew = u + du * self.config.dtp

            # add input state
            u  = ca.MX.sym('u'  + str(k), u_size)
            w += [u]
            lbw += ul
            ubw += uu
            w0 += ug

            # constrain input and delta
            g += [unew - u]
            ubg += [0] * u_size
            lbg += [0] * u_size

            # get new state
            znew = self.F(z,u)

            # penalize input, input delta, and new state
            if k == self.config.N - 1:
                J += ca.bilin(self.config.P,  znew-zref, znew-zref)
            else:
                J += ca.bilin(self.config.Q,  znew-zref, znew-zref)

            J += ca.bilin(self.config.R,  u,         u)
            J += ca.bilin(self.config.dR, du-duref,  du-duref)

            # add variable for next state
            z = ca.MX.sym('z' + str(k+1), z_size)
            w += [z]
            lbw += zl
            ubw += zu
            w0 += zg

            # add dynamics constraint to new state
            g += [z - znew]
            ubg += [0] * z_size
            lbg += [0] * z_size

        prob = {'f':J,'x':ca.vertcat(*w), 'g':ca.vertcat(*g), 'p':ca.vertcat(*p)}
        opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}

        solver_label = 'mpc' if not self.config.use_planar_setup else 'planar_mpc'
        solver = compile_solver(solver_label, 'ipopt', prob, opts)


        self.solver = solver
        self.solver_x0  = w0
        self.solver_lbx = lbw
        self.solver_ubx = ubw
        self.solver_lbg = lbg
        self.solver_ubg = ubg

    def build_mpc_planner(self):
        '''
        sets up a speed planner that will look ahead of the vehicle (fixed intervals of path length)
        and search for a speed that is close to vref but also satisfies normal force requirements
        '''
        fN = self.model.f_N

        z_size = self.model.model_vars.state_dim
        u_size = self.model.model_vars.input_dim

        p = []       # parameters (initial state, cost terms, terminal terms)
        w = []       # states     (vehicle state and inputs)
        w0 = []
        lbw = []
        ubw = []
        J = 0        # cost
        g = []       # nonlinear constraint functions
        lbg = []
        ubg = []

        z0 = ca.MX.sym('z0',z_size)
        u0 = ca.MX.sym('u0',u_size)

        p += [z0]
        p += [u0]

        V = ca.MX.sym('V')
        w += [V]
        ubw += [self.config.V_max]
        lbw += [self.config.V_min]
        w0  += [self.config.vref]

        Vprev = ca.MX.sym('Vprev')  # parameter for rate of change cost on plan
        p += [Vprev]

        J += (V - self.config.vref) ** 2

        # penalize sharp changes in reference velocity
        J += (V - Vprev) **2 * 2000


        for k in range(self.config.planner_N):
            # adjusted state variable with replanned speed and incremented path length
            z = ca.vertcat(z0[0] + self.config.planner_ds*k, 0, 0, V)

            N = fN(z,[0,0,0])
            N = N[0] + N[1] + N[2] + N[3]

            s = ca.MX.sym('s')
            w += [s]
            ubw += [np.inf]
            lbw += [-np.inf]
            w0  += [0]


            g += [N + s]
            ubg += [self.config.Nmax]
            lbg += [self.config.Nmin]

            J += s**2 * 100

        prob = {'f':J,'x':ca.vertcat(*w), 'g':ca.vertcat(*g), 'p':ca.vertcat(*p)}
        opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}

        planner = compile_solver('planner', 'ipopt', prob, opts)


        self.planner = planner
        self.planner_x0  = w0
        self.planner_lbx = lbw
        self.planner_ubx = ubw
        self.planner_lbg = lbg
        self.planner_ubg = ubg
