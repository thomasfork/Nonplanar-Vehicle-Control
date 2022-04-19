import numpy as np
import casadi as ca
from abc import abstractmethod, ABC

from barc3d.pytypes import VehicleConfig, VehicleState, ParametricPose, OrientationQuaternion, BodyForce
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.surfaces.frenet_surface import FrenetSurface
from barc3d.utils.ca_utils import * 
from barc3d.dynamics.integrators import * 

class DynamicsModel(ABC):
    '''
    generic class for 3D vehicle models first introduced in https://arxiv.org/abs/2104.08427
    
    '''
    is_dae = None
    
    def __init__(self, vehicle_config: VehicleConfig, surf: BaseSurface, use_dae = False, use_idas = True):
        self.initialized = False
        
        self.vehicle_config = vehicle_config
        self.surf = surf
        
        self.dt = vehicle_config.dt
        self.m  = vehicle_config.m
        self.g  = vehicle_config.g
        self.lf = vehicle_config.lf
        self.lr = vehicle_config.lr
        self.L  = self.lr + self.lf
        self.tf = vehicle_config.tf
        self.tr = vehicle_config.tr
        self.h  = vehicle_config.h
        self.I1 = vehicle_config.I1
        self.I2 = vehicle_config.I2
        self.I3 = vehicle_config.I3
        self.r  = vehicle_config.tire.r
        self.re = vehicle_config.tire.re
        
        self.use_dae = use_dae # flag for models that can be implemented as a DAE instead of an ODE
        self.use_idas = use_idas # use IDAS for solving ODE or DAE if true, more accurate but prone to getting stuck
        
        self.__post_init__()
        self.__post_init_checks__()
        self.initialized = True
        return
    
    def step(self, state:VehicleState):
        '''
        steps system forwards one time step
        modifies fields of argument "state" by reference
        '''
        if not self.initialized:
            raise TypeError('Dynamics model has not been initialized')
            
        self.coerce_input_limits(state)
        z,u = self.state2zu(state)
        if not self.is_dae:
            zn  = self.f_znew(z, u)
            self.zn2state(state, zn, u)
        else:
            zn, gn = self.f_zgnew(z,u)
            self.zngn2state(state, zn, gn, u)
        state.t += self.dt
        return
        
    def setup_integrator(self, z, u, zdot, param_terms, f_param_terms, pose):
        '''
        utility function for setting up CasADi integrator for dynamics models
        
        arguments;
            z: state vector (SX)
            u: input vector (SX)
            zdot: ode vector (SX)
            param_terms: surface-specific terms that show up in zdot (SX)
            
            ie. zdot = f(z, u, param_terms)
            
            f_param_terms: MX function for interpolated surface-specific terms for the current surface parameterization coefficienets
            pose: argument for f_param_terms, z[0:3]
            
            ie. zdot = f(z, u, f_param_terms(z[0:3])) = f(z,u)
            
        NOTE:
            it is assumed that the convention where the first three elemnents of "z" are the pose, in (s, y, ths)
        '''
        
        self.is_dae = False
        
        zmx = ca.MX.sym('z', z.size())
        umx = ca.MX.sym('u', u.size())
        pose_mx = zmx[0:3]
        
        f_zdot = ca.Function('zdot', [z, u, param_terms], [zdot])
        self.f_param_terms = f_param_terms
        self.f_zdot_full = f_zdot
        
        zdot_mx = f_zdot.call([zmx, umx, f_param_terms(pose_mx)])
        f_zdot_mx  = ca.Function('zdot', [zmx, umx], zdot_mx)
        zdot_mx  = f_zdot_mx (zmx, umx)
        self.f_zdot = f_zdot_mx 
        
        if self.use_idas:
            self.f_znew = idas_integrator(zmx, umx, zdot_mx , self.dt)
        else:
            self.f_znew = collocation_integrator(zmx, umx, f_zdot_mx, self.dt)
        
    
    def setup_dae_integrator(self, z, u, g, zdot, gc, param_terms, f_param_terms, pose):
        
        self.is_dae = True
        
        zmx = ca.MX.sym('z', z.size())
        umx = ca.MX.sym('u', u.size())
        gmx = ca.MX.sym('g', g.size())
        pose_mx = zmx[0:3]
        
        f_zdot = ca.Function('zdot', [z, u, g, param_terms], [zdot])
        self.f_param_terms = f_param_terms
        self.f_zdot_full = f_zdot
        
        zdot_mx = f_zdot.call([zmx, umx, gmx, f_param_terms(pose_mx)])
        f_zdot_mx = ca.Function('zdot', [zmx, umx, gmx], zdot_mx)
        zdot_mx = f_zdot_mx(zmx, umx, gmx)
        self.f_zdot = f_zdot_mx
        
        f_gc = ca.Function('gc', [z, u, g, param_terms], [gc])
        self.f_gc_full = f_gc
        
        gc_mx = f_gc.call([zmx, umx, gmx, f_param_terms(pose_mx)])
        f_gc_mx = ca.Function('gc', [zmx, umx, gmx], gc_mx)
        gc_mx = f_gc_mx(zmx, umx, gmx)
        self.f_gc = f_gc_mx
        
        # set up integrator
        if self.use_idas:
            self.f_znew, self.f_zgnew = idas_dae_integrator(zmx, umx, gmx, zdot_mx, gc_mx, self.dt)
        else:
            self.f_znew, self.f_zgnew = collocation_dae_integrator(zmx, umx, gmx, self.f_zdot, self.f_gc, self.dt, self.vehicle_config)
        
        # set up dae root finder (find g given z and u)
        
        J = ca.bilin(np.eye(gc_mx.shape[0]), gc_mx, gc_mx)
        
        prob = {'f':J,'x':ca.vertcat(gmx), 'p':ca.vertcat(zmx, umx)}
        opts = {'ipopt.print_level': 0, 'ipopt.sb':'yes','print_time':0}
        solver = ca.nlpsol('solver','ipopt',prob, opts)
        
        g_guess = ca.MX.sym('g_guess', gmx.shape)
        sol = solver.call([g_guess, ca.vertcat(zmx, umx),-np.inf,np.inf,0,0,0,0])
        self.f_zu2g = ca.Function('r', [zmx, umx, g_guess], [sol[0]])
        
        return
            
    def sx2mx(self, name, expr, z, u, param_terms, f_param_terms, pose):
        '''
        helper function for converting symbolic expressions (SX) that depend on surface paramterization into
        callable ones for a particular surface.
        
        name - string for the function to return
        expr = (SX) expression to convert
        
        remaining arguments follow syntax of "setup_integrator"
        
        '''
        if isinstance(expr, list):
            func = ca.Function(name, [z,u,param_terms], [*expr])
        else:
            func = ca.Function(name, [z,u,param_terms], [expr])
            
        zmx = ca.MX.sym('z', z.size())
        umx = ca.MX.sym('u', u.size())
        pose_mx = zmx[0:3]
        
        feval = func.call([zmx,umx,f_param_terms(pose_mx)])
        
        func_mx = ca.Function(name, [zmx, umx], feval)
        
        return func_mx
    
    def sx2mx_dae(self, name, expr, z, g, u, param_terms, f_param_terms, pose):
        if isinstance(expr, list):
            func = ca.Function(name, [z,g,u,param_terms], [*expr])
        else:
            func = ca.Function(name, [z,g,u,param_terms], [expr])
        zmx = ca.MX.sym('z', z.size())
        gmx = ca.MX.sym('g', g.size())
        umx = ca.MX.sym('u', u.size())
        pose_mx = zmx[0:3]
    
        feval = func.call([zmx,gmx,umx,f_param_terms(pose_mx)])
        func_mx = ca.Function(name, [zmx, gmx, umx], feval)
        
        return func_mx
        
    def __post_init_checks__(self):
        assert self.is_dae is not None
        
        required_attrs = ['f_N_full', 'f_N']
        for attr in required_attrs:
            assert hasattr(self, attr)
            
        return
    
    @abstractmethod
    def state2zu(self, state: VehicleState):
        ''' convert state object to state vector z and input vector u for the given model'''
        return
    
    @abstractmethod
    def zn2state(self, state: VehicleState, zn, u):
        ''' convert new state vector zn into state object
            should also update fields not part of z, for instance forces '''
        return
    
    def state2du(self, state):
        ''' retrieve input rate from state object in format for given model (primarily for racelines)'''
        return [state.du.a, state.du.y]
    
    @abstractmethod    
    def du2state(self, state, du):
        ''' insert input rate into state object (primarily for racelines) '''
        return
        
    def coerce_input_limits(self, state:VehicleState):
        state.u.a = np.clip(state.u.a, self.vehicle_config.a_min, self.vehicle_config.a_max)
        state.u.y = np.clip(state.u.y, self.vehicle_config.y_min, self.vehicle_config.y_max)
    
    def get_RK4_dynamics(self, dt, steps = 1):
        assert self.is_dae is not None
        
        if self.is_dae:
            return self._get_RK4_dynamics_dae(dt = dt, steps = steps)
        
        z0 = ca.MX.sym('z0',self.f_zdot.size_in(0))
        u0 = ca.MX.sym('u0',self.f_zdot.size_in(1))
        
        dt_s = dt/steps
        z = z0
        
        for j in range(steps):
            k1 = self.f_zdot(z,u0)
            k2 = self.f_zdot(z + dt_s/2*k1, u0)
            k3 = self.f_zdot(z + dt_s/2*k2, u0)
            k4 = self.f_zdot(z + dt_s*  k3, u0)
        
            z  = z  + dt_s / 6 * (k1 + 2*k2 + 2*k3 + k4)
        
        F = ca.Function('F', [z0,u0], [z], ['z0','u'],['zf'])
        
        return F    
    
    def _get_RK4_dynamics_dae(self, dt, steps = 1):
        z0 = ca.MX.sym('z0',self.f_zdot.size_in(0))
        u0 = ca.MX.sym('u0',self.f_zdot.size_in(1))
        g0 = ca.MX.sym('g0',self.f_zdot.size_in(2))
    
        dt_s = dt/steps
        z = z0
        
        for j in range(steps):
            k1 = self.f_zdot(z,u0,g0)
            k2 = self.f_zdot(z + dt_s/2*k1, u0, g0)
            k3 = self.f_zdot(z + dt_s/2*k2, u0, g0)
            k4 = self.f_zdot(z + dt_s*  k3, u0, g0)
        
            z  = z  + dt_s / 6 * (k1 + 2*k2 + 2*k3 + k4)
        
        F = ca.Function('F', [z0,u0,g0], [z], ['z0','u','g'],['zf'])
        return F
    
    def _eval_vel(self, v1, v2, w3):
        ''' compute standard results from vehicle velocity for any model '''
        if not self.surf.initialized:
            raise TypeError('Chosen surface is not initialized')
        sym_rep = self.surf.sym_rep
                
        if self.vehicle_config.road_surface:
            n = self.h
        else:
            n = 0
            
        # parametric position derivative
        one = sym_rep['one']
        two = sym_rep['two']
        J   = sym_rep['J']
        dsdy = ca.inv(one - n * two) @ J @ ca.vertcat(v1,v2)
        s_dot = dsdy[0]
        y_dot = dsdy[1]
            
        # parametric orientation derivative
        ws = sym_rep['ws']
        wy = sym_rep['wy']
        ths_dot = w3 + ws * s_dot + wy * y_dot
            
        # constrained angular velocity terms
        w2w1 = ca.inv(J) @ two @ dsdy
        w2 = -w2w1[0]
        w1 = w2w1[1]
            
        # constrained linear velocity terms
        v3 = 0
        v3_dot = 0
        
        # drag forces
        Fd1 = 0 -self.vehicle_config.c1 * v1 * ca_abs(v1)
        Fd2 = 0 -self.vehicle_config.c2 * v2 * ca_abs(v2)
        Fd3 = 0 -self.vehicle_config.c3 * v1 * v1
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0
            
        # gravity forces
        R = sym_rep['R']
        e1 = R[:,0]
        e2 = R[:,1]
        e3 = R[:,2]
        Fg1 = -self.m * self.g * e1[2]
        Fg2 = -self.m * self.g * e2[2]
        Fg3 = -self.m * self.g * e3[2]
            
        # net normal force
        N_tot = self.m * (v3_dot + v2*w1 - v1*w2) - Fd3 - Fg3 
            
        return {'s_dot': s_dot,
                'y_dot': y_dot,
                'ths_dot': ths_dot,
                'v3': v3,
                'w1': w1,
                'w2': w2,
                'Fd1': Fd1,
                'Fd2': Fd2,
                'Fd3': Fd3,
                'Kd1': Kd1,
                'Kd2': Kd2,
                'Kd3': Kd3,
                'Fg1': Fg1,
                'Fg2': Fg2,
                'Fg3': Fg3,
                'N_tot': N_tot}
                
    # functions to give bounds for racelines, planning, etc...
    @abstractmethod
    def zu(self):
        return
        
    @abstractmethod
    def zl(self):
        return
    
    @abstractmethod
    def uu(self):
        return
    
    @abstractmethod
    def ul(self):
        return
    
    @abstractmethod
    def duu(self):
        return
    
    @abstractmethod
    def dul(self):
        return

    @abstractmethod
    def _add_model_stage_constraints(self, zkj, ukj, g, lbg, ubg, param_terms = None):
        '''
        function for adding stage constraints - provide param terms if they are fixed, otherwise they are implicityly computed from zkj
        
        for dae models use (zjk, ukj, gkj, g, lbg, ubg, param_tersm = None)
        '''
        return
        
class KinematicBicycle3D(DynamicsModel):
    '''
    Nonplanar kinematic bicycle model 
    '''             
    def __post_init__(self):
        if not self.surf.initialized:
            raise TypeError('Chosen surface is not initialized')
        
        config = self.vehicle_config
        sym_rep = self.surf.sym_rep
        
        # state
        s   = sym_rep['s']
        y   = sym_rep['y']
        ths = sym_rep['ths']
        v   = ca.SX.sym('v')
        
        # input
        uy = ca.SX.sym('uy')
        ua = ca.SX.sym('ua')
        
        # parameterization terms
        pose = sym_rep['pose']
        param_terms = sym_rep['param_terms']
        f_param_terms = sym_rep['f_param_terms']
        
        # velocity components
        beta = ca.atan(config.lr / (config.lf + config.lr) * ca.tan(uy))
        v1 = ca.cos(beta) * v
        v2 = ca.sin(beta) * v
        w3 = v * ca.cos(beta) / (config.lr + config.lf) * ca.tan(uy)
        
        ev = self._eval_vel(v1,v2,w3)
        
        
        # acceleration due to drag
        ad = (ev['Fd1'] * ca.cos(beta) + ev['Fd2'] * ca.sin(beta)) / self.m
        
        # acceleration due to gravity
        ag = (ev['Fg1'] * ca.cos(beta) + ev['Fg2'] * ca.sin(beta)) / self.m
        
        # velocity state evolution
        v_dot = ua + ag + ad
        
        # state evolution
        z = ca.vertcat(s,y,ths,v)
        u = ca.vertcat(ua,uy)
        z_dot = ca.vertcat(ev['s_dot'], ev['y_dot'], ev['ths_dot'], v_dot)
        self.setup_integrator(z, u, z_dot, param_terms, f_param_terms, pose)
        
        # auxilliary functions:
        # lateral acceleration needed to oppose gravity
        agt = -(-ev['Fg1'] * ca.sin(beta) + ev['Fg2'] * ca.cos(beta)) / self.m
        self.f_agt_full = ca.Function('agt', [z, u, param_terms], [agt])
        self.f_agt      = self.sx2mx('agt', agt, z, u, param_terms, f_param_terms, pose)
        
        # net normal force
        N = ev['N_tot']
        self.f_N_full = ca.Function('N',[z, u, param_terms],[N])
        self.f_N = self.sx2mx('N', N, z, u, param_terms, f_param_terms, pose)
        return
    
    def state2zu(self, state:VehicleState):
        z = [state.p.s, state.p.y, state.p.ths, state.v.signed_mag()]
        u = [state.u.a, state.u.y]
        return z,u
    
    def zn2state(self, state, zn, u):
        state.u.a = u[0].__float__()
        state.u.y = u[1].__float__()
        
        beta = np.arctan(self.lr/(self.lf+self.lr)*np.tan(state.u.y))
            
        state.p.s   = zn[0].__float__()
        state.p.y   = zn[1].__float__()
        state.p.ths = zn[2].__float__() 
        state.v.v1 = zn[3].__float__() * np.cos(beta)
        state.v.v2 = zn[3].__float__() * np.sin(beta)
        state.w.w3 = state.v.v1 / (self.lf + self.lr) * np.tan(state.u.y)
            
        state.p.n = self.h if self.vehicle_config.road_surface else 0
        
        self.surf.frenet_to_global(state)
        
        state.fb.f3 = self.f_N(zn,u).__float__()
        state.hw.wfr = state.v.v1 / self.re
        state.hw.wfl = state.v.v1 / self.re
        state.hw.wrr = state.v.v1 / self.re
        state.hw.wrl = state.v.v1 / self.re
        return
        
    def du2state(self, state, du):
        state.du.a = du[0].__float__()
        state.du.y = du[1].__float__()
        
    def zu(self):
        return [self.surf.s_max(),  self.surf.y_max(), np.inf, np.inf]
        
    def zl(self):
        return [self.surf.s_min(), self.surf.y_min(), -np.inf,-np.inf]
    
    def uu(self):
        return [self.vehicle_config.a_max, self.vehicle_config.y_max]
    
    def ul(self):
        return [self.vehicle_config.a_min, self.vehicle_config.y_min]
    
    def duu(self):
        return [self.vehicle_config.da_max, self.vehicle_config.dy_max]
    
    def dul(self):
        return [self.vehicle_config.da_min, self.vehicle_config.dy_min]
    
    def _add_model_stage_constraints(self, zkj, ukj, g, lbg, ubg, param_terms = None):
        if param_terms is None:
            N = self.f_N(zkj, ukj)
            agt = self.f_agt(zkj, ukj)
        else:
            N = self.f_N_full(zkj, ukj, param_terms)
            agt = self.f_agt_full(zkj, ukj, param_terms)
        
        g += [N / self.vehicle_config.N_max ]
        ubg += [1]
        lbg += [0]
        
        allowed_accel = (N * self.vehicle_config.tire.mu / self.m) 
        allowed_accel = allowed_accel **2 
        used_accel = (zkj[3]**2 * ukj[1] / self.L + agt)**2 + ukj[0]**2
        g += [ used_accel / allowed_accel ]
        ubg += [ 1]
        lbg += [-np.inf]
        return g, ubg, lbg
    
class KinematicBicyclePlanar(KinematicBicycle3D):
    '''
    planar kinematic vehicle model for https://arxiv.org/abs/2104.08427
    not meant for simulation, only numerical implementation of a planar kinematic model on a nonplanar surface
    
    
    
    '''
    def __post_init__(self):
        if not self.surf.initialized:
            raise TypeError('Chosen surface is not initialized')
        
        config = self.vehicle_config
        sym_rep = self.surf.sym_rep
        
        if 'da' in sym_rep:
            kappa = sym_rep['da']
        elif isinstance(self.surf, FrenetSurface):
            kappa = sym_rep['k']
        else:
            raise NotImplementedError('Unable to extract curvature from surface %s'%type(self.surf).__str__)
        
        # state
        s   = sym_rep['s']
        y   = sym_rep['y']
        ths = sym_rep['ths']
        v   = ca.SX.sym('v')
        
        # input
        uy = ca.SX.sym('uy')
        ua = ca.SX.sym('ua')
        
        # parameterization terms
        pose = sym_rep['pose']
        param_terms = sym_rep['param_terms']
        f_param_terms = sym_rep['f_param_terms']
        
        # velocity components
        beta = ca.atan(config.lr / (config.lf + config.lr) * ca.tan(uy))
        v1 = ca.cos(beta) * v
        v2 = ca.sin(beta) * v
        w3 = v * ca.cos(beta) / (config.lr + config.lf) * ca.tan(uy)
        
        # pose evolution
        s_dot = (v1 * ca.cos(ths) - v2 * ca.sin(ths)) / (1- kappa * y)
        y_dot =  v1 * ca.sin(ths) + v2 * ca.cos(ths)
        ths_dot = w3 - kappa * s_dot
        
        # velocity evolution
        v_dot = ua
        
        # state evolution
        z = ca.vertcat(s,y,ths,v)
        u = ca.vertcat(ua,uy)
        z_dot = ca.vertcat(s_dot, y_dot, ths_dot, v_dot)         
        self.setup_integrator(z, u, z_dot, param_terms, f_param_terms, pose)
        
        # net normal force
        N = config.m * config.g
        self.f_N_full = ca.Function('N',[z, u, param_terms],[N])
        self.f_N = self.sx2mx('N', N, z, u, param_terms, f_param_terms, pose)
        
        
        # transverse gravity acceleration, for friction cone constraints
        agt = 0
        self.f_agt_full = ca.Function('agt', [z, u, param_terms], [agt])
        self.f_agt      = self.sx2mx('agt', agt, z, u, param_terms, f_param_terms, pose)
        
        return
    
    def step(self, state:VehicleState):
        raise TypeError('Simulating with planar-only model within 3D dynamics is invalid')


class DynamicBicycle3D(DynamicsModel):
    '''
    Nonplanar dynamic bicycle with static weight distribution and simplified longitudinal dynamics
    '''
    def __post_init__(self):
        if not self.surf.initialized:
            raise TypeError('Chosen surface is not initialized')
    
        config = self.vehicle_config
        sym_rep = self.surf.sym_rep
        
        if self.vehicle_config.road_surface:
            n = self.h
        else:
            n = 0

        # state
        s   = sym_rep['s']
        y   = sym_rep['y']
        ths = sym_rep['ths']
        v1   = ca.SX.sym('v1')
        v2   = ca.SX.sym('v2')
        w3   = ca.SX.sym('w3')

        #input
        uy = ca.SX.sym('uy')
        ua = ca.SX.sym('ua')
        
        # parameterization terms
        pose = sym_rep['pose']
        param_terms = sym_rep['param_terms']
        f_param_terms = sym_rep['f_param_terms']
        
        ev = self._eval_vel(v1,v2,w3)
        
        v3 = ev['v3']
        w1 = ev['w1']
        w2 = ev['w2']
        
        # quasistatic loading
        Nf = self.lr / self.L * ev['N_tot']
        Nr = self.lf / self.L * ev['N_tot']
        
        # tire forces
        alpha_f, F1_f, F2_f = self.vehicle_config.tire.tire_model_lateral(uy, Nf, v1, v2, v3, w1, w2, w3,  self.lf, 0, -self.h + self.r)
        alpha_r, F1_r, F2_r = self.vehicle_config.tire.tire_model_lateral(0 , Nr, v1, v2, v3, w1, w2, w3, -self.lr, 0, -self.h + self.r)
        
        # net tire force and torque
        Ft1 = F1_f + F1_r
        Ft2 = F2_f + F2_r 
        Kt3 = self.lf * F2_f - self.lr * F2_r
        
        # net force and torque on body
        F1 = Ft1 + ev['Fg1'] + ev['Fd1']
        F2 = Ft2 + ev['Fg2'] + ev['Fd2']
        K3 = Kt3 + ev['Kd3']
        
        # velocity evolution 
        v1_dot = w3*v2 + F1 / self.m + ua
        v2_dot =-w3*v1 + F2 / self.m
        w3_dot = (K3 - (self.I2 - self.I1) * w1*w2) / self.I3
        
        z = ca.vertcat(s,y,ths,v1,v2,w3)
        u = ca.vertcat(ua,uy)
            
        z_dot = ca.vertcat(ev['s_dot'], ev['y_dot'], ev['ths_dot'], v1_dot, v2_dot, w3_dot)
            
        self.setup_integrator(z, u, z_dot, param_terms, f_param_terms, pose)
        
        # helper functions
        self.f_alpha = self.sx2mx('alpha', [alpha_f, alpha_r], z, u, param_terms, f_param_terms, pose)
        self.f_N_full = ca.Function('N',[z, u, param_terms],[ev['N_tot']])
        self.f_N      = self.sx2mx('N', [ev['N_tot']], z, u, param_terms, f_param_terms, pose)
        self.f_NfNr   = self.sx2mx('N', [Nf, Nr], z, u, param_terms, f_param_terms, pose)
        return

        
    def state2zu(self, state:VehicleState):
        z = [state.p.s, state.p.y, state.p.ths, state.v.v1, state.v.v2, state.w.w3]
        u = [state.u.a, state.u.y]
        
        return z,u
    
    def zn2state(self, state, zn, u):
        state.u.a = u[0].__float__()
        state.u.y = u[1].__float__()
        
        state.p.s   = zn[0].__float__()
        state.p.y   = zn[1].__float__()
        state.p.ths = zn[2].__float__() 
        state.v.v1  = zn[3].__float__() 
        state.v.v2  = zn[4].__float__() 
        state.w.w3  = zn[5].__float__() 
        
        state.p.n = self.h if self.vehicle_config.road_surface else 0
        
        self.surf.frenet_to_global(state)
        
        N = self.f_NfNr(zn, u)
        alpha = self.f_alpha(zn,u)
        
        state.tfr.a = alpha[0].__float__()
        state.tfl.a = alpha[0].__float__()
        state.tfr.N = N[0].__float__() / 2
        state.tfl.N = N[0].__float__() / 2
        
        state.trr.a = alpha[1].__float__()
        state.trl.a = alpha[1].__float__()
        state.trr.N = N[1].__float__() / 2
        state.trl.N = N[1].__float__() / 2
        
        state.hw.wfr = state.v.v1 / self.re
        state.hw.wfl = state.v.v1 / self.re
        state.hw.wrr = state.v.v1 / self.re
        state.hw.wrl = state.v.v1 / self.re
        
        state.fb.f3 = N[0].__float__() + N[1].__float__()
        return
        
    def du2state(self, state, du):
        state.du.a = du[0].__float__()
        state.du.y = du[1].__float__()
        
    def zu(self):
        return [self.surf.s_max(),  self.surf.y_max(), np.inf, np.inf, np.inf, np.inf]
        
    def zl(self):
        return [self.surf.s_min(), self.surf.y_min(), -np.inf, -np.inf, -np.inf, -np.inf]
    
    def uu(self):
        return [self.vehicle_config.a_max, self.vehicle_config.y_max]
    
    def ul(self):
        return [self.vehicle_config.a_min, self.vehicle_config.y_min]
    
    def duu(self):
        return [self.vehicle_config.da_max, self.vehicle_config.dy_max]
    
    def dul(self):
        return [self.vehicle_config.da_min, self.vehicle_config.dy_min]
        
    def _add_model_stage_constraints(self, zkj, ukj, g, lbg, ubg, param_terms = None):
        if param_terms is None:
            N = self.f_N(zkj, ukj)
        else:
            N = self.f_N_full(zkj, ukj, param_terms)
            
        g += [N / self.vehicle_config.N_max ]
        ubg += [1]
        lbg += [0]
        
        allowed_accel = self.vehicle_config.tire.mu * N/ self.vehicle_config.m
        used_accel = ukj[0]
        g += [ used_accel / allowed_accel ]
        ubg += [ 1]
        lbg += [-1]
        return g, ubg, lbg
        
        
class DynamicTwoTrack3D(DynamicsModel):
    '''
    Nonplanar dynamic two-track model with dynamic weight distribution and tire angular velocity states
    
    Not recommended for use, buggy and numerically unpleasant
    '''
    def __post_init__(self):
        if not self.surf.initialized:
            raise TypeError('Chosen surface is not initialized')
        
        config = self.vehicle_config
        sym_rep = self.surf.sym_rep
        
        if self.vehicle_config.road_surface:
            n = self.h
        else:
            n = 0
        
        # state
        s   = sym_rep['s']
        y   = sym_rep['y']
        ths = sym_rep['ths']
        v1   = ca.SX.sym('v1')
        v2   = ca.SX.sym('v2')
        w3   = ca.SX.sym('w3')
        wfr = ca.SX.sym('wfr')
        wfl = ca.SX.sym('wfl')
        wrr = ca.SX.sym('wrr')
        wrl = ca.SX.sym('wrl')
        
        #input
        uy = ca.SX.sym('uy')
        ua = ca.SX.sym('ua')
        
        # parameterization terms
        pose = sym_rep['pose']
        param_terms = sym_rep['param_terms']
        f_param_terms = sym_rep['f_param_terms']
        
        ev = self._eval_vel(v1,v2,w3)
        v3 = ev['v3']
        w1 = ev['w1']
        w2 = ev['w2']
        
        if not self.use_dae:
            Nfr = ca.SX.sym('Nfr')
            Nfl = ca.SX.sym('Nfl')
            Nrr = ca.SX.sym('Nrr')
            Nrl = ca.SX.sym('Nrl')
        else:
            # DAE state
            Nf    = ca.SX.sym('Nf')
            Nr    = ca.SX.sym('Nr')
            Delta = ca.SX.sym('Delta')
            
            # tire normal forces        
            Nfr = Nf/2 - self.tf * Delta
            Nfl = Nf/2 + self.tf * Delta
            Nrr = Nr/2 - self.tr * Delta
            Nrl = Nr/2 + self.tr * Delta
        
        # tire steering angles (Ackermann Steering)
        y_fr = ca.arctan(ca.tan(uy) / (-self.tf/2/self.L * ca.tan(uy) + 1))
        y_fl = ca.arctan(ca.tan(uy) / ( self.tf/2/self.L * ca.tan(uy) + 1))
        y_rr = 0
        y_rl = 0
        
        # tire torques 
        Tfr = self.m * ua * self.re / 4
        Tfl = self.m * ua * self.re / 4
        Trr = self.m * ua * self.re / 4
        Trl = self.m * ua * self.re / 4
        
        # tire slip ratio, slip angle, forces, and angular acceleration
        sigma_fr, alpha_fr, F1_fr, F2_fr, wfr_dot = self.vehicle_config.tire.tire_model(wfr, y_fr, Tfr, Nfr, v1, v2, v3, w1, w2, w3,  self.lf, -self.tf, -self.h + self.r)
        sigma_fl, alpha_fl, F1_fl, F2_fl, wfl_dot = self.vehicle_config.tire.tire_model(wfl, y_fl, Tfl, Nfl, v1, v2, v3, w1, w2, w3,  self.lf,  self.tf, -self.h + self.r)
        sigma_rr, alpha_rr, F1_rr, F2_rr, wrr_dot = self.vehicle_config.tire.tire_model(wrr, y_rr, Trr, Nrr, v1, v2, v3, w1, w2, w3, -self.lr, -self.tr, -self.h + self.r)
        sigma_rl, alpha_rl, F1_rl, F2_rl, wrl_dot = self.vehicle_config.tire.tire_model(wrl, y_rl, Trl, Nrl, v1, v2, v3, w1, w2, w3, -self.lr,  self.tr, -self.h + self.r)
        
        # net tire force and torque
        Ft1 = F1_fr + F1_fl + F1_rr + F1_rl 
        Ft2 = F2_fr + F2_fl + F2_rr + F2_rl
        Kt3 = self.lf * (F2_fr + F2_fl) - self.lr * (F2_rr + F2_rl) \
            + self.tf * (F1_fr - F1_fl) + self.tr * (F1_rr - F1_rl)
        
        # net force and torque on body
        F1 = Ft1 + ev['Fg1'] + ev['Fd1']
        F2 = Ft2 + ev['Fg2'] + ev['Fd2']
        K3 = Kt3 + ev['Kd3']
        
        # velocity evolution 
        v1_dot = w3*v2 + F1 / self.m
        v2_dot =-w3*v1 + F2 / self.m
        v3_dot = 0
        w3_dot = (K3 - (self.I2 - self.I1) * w1*w2) / self.I3
        
        # approximate angular acceleration
        one = sym_rep['one']
        two = sym_rep['two']
        J = sym_rep['J']
        dw1w2 = ca.inv(J) @ two @ ca.inv(one - n * two) @ J @ ca.vertcat(v1_dot,v2_dot)
        w2_dot = -dw1w2[0]
        w1_dot =  dw1w2[1]
        
        # required forces and torques
        F3_required = self.m * (v3_dot + v2*w1 - v1*w2)
        K1_required = self.I1 * w1_dot + (self.I3 - self.I2)*w2*w3
        K2_required = self.I2 * w2_dot + (self.I1 - self.I3)*w3*w1
        
        # required forces and torques from only the tire normal forces
        F3_N_required = F3_required - ev['Fd3'] - ev['Fg3']
        K1_N_required = K1_required - ev['Kd1'] - self.h * (Ft2)
        K2_N_required = K2_required - ev['Kd2'] + self.h * (Ft1)
        
        # compute resulting Nf, Nr, and Delta to enforce consistency
        Nf_required = self.lr / self.L * F3_N_required - 1/self.L * K2_N_required
        Nr_required = self.lf / self.L * F3_N_required + 1/self.L * K2_N_required
        Delta_required = 1/2/(self.tf**2 + self.tr**2) * K1_N_required
        
        
        if not self.use_dae:
            Nfr_eq = Nf_required/2 - self.tf * Delta_required
            Nfl_eq = Nf_required/2 + self.tf * Delta_required
            Nrr_eq = Nr_required/2 - self.tr * Delta_required
            Nrl_eq = Nr_required/2 + self.tr * Delta_required
            
            Nfr_dot = (Nfr_eq - Nfr) * 10
            Nfl_dot = (Nfl_eq - Nfl) * 10
            Nrr_dot = (Nrr_eq - Nrr) * 10
            Nrl_dot = (Nrl_eq - Nrl) * 10
            
            z = ca.vertcat(s,y,ths,v1,v2,w3,wfr,wfl,wrr,wrl,Nfr, Nfl, Nrr, Nrl)
            u = ca.vertcat(ua,uy)
            
            z_dot = ca.vertcat(ev['s_dot'], ev['y_dot'], ev['ths_dot'], v1_dot, v2_dot, w3_dot, 
                               wfr_dot, wfl_dot, wrr_dot, wrl_dot, Nfr_dot, Nfl_dot, Nrr_dot, Nrl_dot)
            
            self.setup_integrator(z, u, z_dot, param_terms, f_param_terms, pose)
            
            
            # helper functions
            self.f_N_full = ca.Function('N', [z, u, param_terms], [ca.vertcat(Nfr, Nfl, Nrr, Nrl)])
            self.f_N = self.sx2mx('N', [Nfr, Nfl, Nrr, Nrl],     z, u, param_terms, f_param_terms, pose)
            self.f_y = self.sx2mx('y', [y_fr, y_fl, y_rr, y_rl], z, u, param_terms, f_param_terms, pose)
            self.f_T = self.sx2mx('T', [Tfr, Tfl, Trr, Trl],     z, u, param_terms, f_param_terms, pose)
            self.f_sigma = self.sx2mx('sigma', [sigma_fr, sigma_fl, sigma_rr, sigma_rl], z, u, param_terms, f_param_terms, pose)
            self.f_alpha = self.sx2mx('alpha', [alpha_fr, alpha_fl, alpha_rr, alpha_rl], z, u, param_terms, f_param_terms, pose)
        else:
            
            z = ca.vertcat(s,y,ths,v1,v2,w3,wfr,wfl,wrr,wrl)
            u = ca.vertcat(ua,uy)
            g = ca.vertcat(Nf, Nr, Delta)
            
            z_dot = ca.vertcat(ev['s_dot'], ev['y_dot'], ev['ths_dot'], v1_dot, v2_dot, w3_dot, wfr_dot, wfl_dot, wrr_dot, wrl_dot)
            dae = ca.vertcat(Nf - Nf_required, Nr - Nr_required, Delta - Delta_required)
             
            self.setup_dae_integrator(z, u, g, z_dot, dae, param_terms, f_param_terms, pose)
            
            # helper functions
            self.f_N_full = ca.Function('N', [z, u, g, param_terms], [ca.vertcat(Nfr, Nfl, Nrr, Nrl)])
            self.f_N = self.sx2mx_dae('N', [Nfr, Nfl, Nrr, Nrl], z, g, u, param_terms, f_param_terms, pose)
            self.f_y = self.sx2mx_dae('y', [y_fr, y_fl, y_rr, y_rl], z, g, u, param_terms, f_param_terms, pose)
            self.f_T = self.sx2mx_dae('T', [Tfr, Tfl, Trr, Trl], z, g, u, param_terms, f_param_terms, pose)
            self.f_sigma = self.sx2mx_dae('sigma', [sigma_fr, sigma_fl, sigma_rr, sigma_rl], z, g, u, param_terms, f_param_terms, pose)
            self.f_alpha = self.sx2mx_dae('alpha', [alpha_fr, alpha_fl, alpha_rr, alpha_rl], z, g, u, param_terms, f_param_terms, pose)
        
    def state2zu(self, state: VehicleState):
        if not self.use_dae:
            z = [state.p.s, state.p.y, state.p.ths, state.v.v1, state.v.v2, state.w.w3, state.hw.wfr, state.hw.wfl, state.hw.wrr, state.hw.wrl, state.tfr.N,
                 state.tfl.N, state.trr.N, state.trl.N]
        else:
            z = [state.p.s, state.p.y, state.p.ths, state.v.v1, state.v.v2, state.w.w3, state.hw.wfr, state.hw.wfl, state.hw.wrr, state.hw.wrl]
        u = [state.u.a, state.u.y]
        return z,u
        
    def state2zug(self, state: VehicleState, g0 = None):
        assert self.use_dae
        
        z = [state.p.s, state.p.y, state.p.ths, state.v.v1, state.v.v2, state.w.w3, state.hw.wfr, state.hw.wfl, state.hw.wrr, state.hw.wrl]
        u = [state.u.a, state.u.y]
        
        if g0 is None:
            g0 = [self.m * self.g * self.lr / self.L,
                  self.m * self.g * self.lf / self.L,
                  0]
        
        g = self.f_zu2g(z, u, g0)
        g = np.array(g).squeeze().tolist()
        
        return z, u, g
    
    def zn2state(self, state, zn, u): 
        state.u.a = u[0].__float__()
        state.u.y = u[1].__float__()
        
        state.p.s   = zn[0].__float__()
        state.p.y   = zn[1].__float__()
        state.p.ths = zn[2].__float__() 
        state.v.v1  = zn[3].__float__() 
        state.v.v2  = zn[4].__float__() 
        state.w.w3  = zn[5].__float__() 
        state.hw.wfr= zn[6].__float__() 
        state.hw.wfl= zn[7].__float__() 
        state.hw.wrr= zn[8].__float__() 
        state.hw.wrl= zn[9].__float__() 
        
        state.p.n = self.h if self.vehicle_config.road_surface else 0
        
        if not self.use_dae:
            state.tfr.N = zn[10].__float__()
            state.tfl.N = zn[11].__float__()
            state.trr.N = zn[12].__float__()
            state.trl.N = zn[13].__float__()
            
            y = self.f_y(zn, u)
            sigma = self.f_sigma(zn, u)
            alpha = self.f_alpha(zn, u)

            state.tfr.y = y[0].__float__()
            state.tfr.a = alpha[0].__float__()
            state.tfr.s = sigma[0].__float__()
            
            state.tfl.y = y[1].__float__()
            state.tfl.a = alpha[1].__float__()
            state.tfl.s = sigma[1].__float__()
            
            state.trr.y = y[2].__float__()
            state.trr.a = alpha[2].__float__()
            state.trr.s = sigma[2].__float__()
            
            state.trl.y = y[3].__float__()
            state.trl.a = alpha[3].__float__()
            state.trl.s = sigma[3].__float__()
            
            state.fb.f3 = state.tfr.N + state.tfl.N + state.trr.N + state.trl.N
            
        self.surf.frenet_to_global(state)
        return
    
    def zngn2state(self, state, zn, gn, u):
        self.zn2state(state, zn, u)
        
        N = self.f_N(zn, gn, u)
        y = self.f_y(zn, gn, u)
        sigma = self.f_sigma(zn, gn, u)
        alpha = self.f_alpha(zn, gn, u)

        state.tfr.y = y[0].__float__()
        state.tfr.a = alpha[0].__float__()
        state.tfr.s = sigma[0].__float__()
        state.tfr.N = N[0].__float__()
        
        state.tfl.y = y[1].__float__()
        state.tfl.a = alpha[1].__float__()
        state.tfl.s = sigma[1].__float__()
        state.tfl.N = N[1].__float__()
        
        state.trr.y = y[2].__float__()
        state.trr.a = alpha[2].__float__()
        state.trr.s = sigma[2].__float__()
        state.trr.N = N[2].__float__()
        
        state.trl.y = y[3].__float__()
        state.trl.a = alpha[3].__float__()
        state.trl.s = sigma[3].__float__()
        state.trl.N = N[3].__float__()
        
        state.fb.f3 = N[0] + N[1] + N[2] + N[3]
        return
        
    def du2state(self, state, du):
        state.du.a = du[0].__float__()
        state.du.y = du[1].__float__()
        
    def zu(self):
        return [self.surf.s_max(),  self.surf.y_max(), np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        
    def zl(self):
        return [self.surf.s_min(), self.surf.y_min(), -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    
    def uu(self):
        return [self.vehicle_config.a_max, self.vehicle_config.y_max]
    
    def ul(self):
        return [self.vehicle_config.a_min, self.vehicle_config.y_min]
    
    def duu(self):
        return [self.vehicle_config.da_max, self.vehicle_config.dy_max]
    
    def dul(self):
        return [self.vehicle_config.da_min, self.vehicle_config.dy_min]
        
    def _add_model_stage_constraints(self, zkj, ukj, gkj, g, lbg, ubg, param_terms = None):
        if param_terms is None:
            N = self.f_N(zkj, ukj, gkj)
        else:
            N = self.f_N_full(zkj, ukj, gkj, param_terms)
        g += [N / self.vehicle_config.N_max]
        ubg += [0.5] * 4
        lbg += [0.] * 4
        return g, ubg, lbg
        
        
class DynamicTwoTrackSlipInput3D(DynamicsModel):
    '''
    Nonplanar dynamic two-track model with dynamic weight distribution and tire slip ratio treated as an input
    - always set up as a DAE
    '''        
    def __post_init__(self):
        if not self.surf.initialized:
            raise TypeError('Chosen surface is not initialized')
        self.use_dae = True
        
        config = self.vehicle_config
        sym_rep = self.surf.sym_rep
        
        if self.vehicle_config.road_surface:
            n = self.h
        else:
            n = 0
        
        # state
        s   = sym_rep['s']
        y   = sym_rep['y']
        ths = sym_rep['ths']
        v1   = ca.SX.sym('v1')
        v2   = ca.SX.sym('v2')
        w3   = ca.SX.sym('w3')
        
        # dae state
        Nf    = ca.SX.sym('Nf')
        Nr    = ca.SX.sym('Nr')
        Delta = ca.SX.sym('Delta')
        
        # tire normal forces        
        Nfr = Nf/2 - self.tf * Delta
        Nfl = Nf/2 + self.tf * Delta
        Nrr = Nr/2 - self.tr * Delta
        Nrl = Nr/2 + self.tr * Delta
        
        #input
        uy = ca.SX.sym('uy')
        sigma_fr = ca.SX.sym('sigma_fr')
        sigma_fl = ca.SX.sym('sigma_fl')
        sigma_rr = ca.SX.sym('sigma_rr')
        sigma_rl = ca.SX.sym('sigma_rl')
        
        # parameterization terms
        pose = sym_rep['pose']
        param_terms = sym_rep['param_terms']
        f_param_terms = sym_rep['f_param_terms']
        
        ev = self._eval_vel(v1,v2,w3)
        v3 = ev['v3']
        w1 = ev['w1']
        w2 = ev['w2']

        # tire steering angles (Ackermann Steering)
        y_fr = ca.arctan(ca.tan(uy) / (-self.tf/2/self.L * ca.tan(uy) + 1))
        y_fl = ca.arctan(ca.tan(uy) / ( self.tf/2/self.L * ca.tan(uy) + 1))
        y_rr = 0
        y_rl = 0
        
        # tire force model
        alpha_fr, F1_fr, F2_fr = self.vehicle_config.tire.tire_model_sigma_input(sigma_fr, y_fr, Nfr, v1,v2,v3,w1,w2,w3,  self.lf, -self.tf, -self.h + self.r)
        alpha_fl, F1_fl, F2_fl = self.vehicle_config.tire.tire_model_sigma_input(sigma_fl, y_fl, Nfl, v1,v2,v3,w1,w2,w3,  self.lf,  self.tf, -self.h + self.r)
        alpha_rr, F1_rr, F2_rr = self.vehicle_config.tire.tire_model_sigma_input(sigma_rr, y_rr, Nrr, v1,v2,v3,w1,w2,w3, -self.lr, -self.tr, -self.h + self.r)
        alpha_rl, F1_rl, F2_rl = self.vehicle_config.tire.tire_model_sigma_input(sigma_rl, y_rl, Nrl, v1,v2,v3,w1,w2,w3, -self.lr,  self.tr, -self.h + self.r)
        
        # net tire force and torque
        Ft1 = F1_fr + F1_fl + F1_rr + F1_rl 
        Ft2 = F2_fr + F2_fl + F2_rr + F2_rl
        Kt3 = self.lf * (F2_fr + F2_fl) - self.lr * (F2_rr + F2_rl) \
            + self.tf * (F1_fr - F1_fl) + self.tr * (F1_rr - F1_rl)
        
        # net force and torque on body
        F1 = Ft1 + ev['Fg1'] + ev['Fd1']
        F2 = Ft2 + ev['Fg2'] + ev['Fd2']
        K3 = Kt3 + ev['Kd3']
        
        # velocity evolution 
        v1_dot = w3*v2 + F1 / self.m 
        v2_dot =-w3*v1 + F2 / self.m
        v3_dot = 0
        w3_dot = (K3 - (self.I2 - self.I1) * w1*w2) / self.I3
        
        # approximate angular acceleration
        one = sym_rep['one']
        two = sym_rep['two']
        J = sym_rep['J']
        dw1w2 = ca.inv(J) @ two @ ca.inv(one - n * two) @ J @ ca.vertcat(v1_dot,v2_dot)
        w2_dot = -dw1w2[0]
        w1_dot =  dw1w2[1]
        
        # required forces and torques
        F3_required = self.m * (v3_dot + v2*w1 - v1*w2)
        K1_required = self.I1 * w1_dot + (self.I3 - self.I2)*w2*w3
        K2_required = self.I2 * w2_dot + (self.I1 - self.I3)*w3*w1
        
        # required forces and torques from only the tire normal forces
        F3_N_required = F3_required - ev['Fd3'] - ev['Fg3']
        K1_N_required = K1_required - ev['Kd1'] - self.h * (Ft2)
        K2_N_required = K2_required - ev['Kd2'] + self.h * (Ft1)
        
        # compute resulting Nf, Nr, and Delta to enforce consistency
        Nf_required = self.lr / self.L * F3_N_required - 1/self.L * K2_N_required
        Nr_required = self.lf / self.L * F3_N_required + 1/self.L * K2_N_required
        Delta_required = 1/2/(self.tf**2 + self.tr**2) * K1_N_required
        
        z = ca.vertcat(s,y,ths,v1,v2,w3)
        u = ca.vertcat(sigma_fr, sigma_fl, sigma_rr, sigma_rl, uy)
        g = ca.vertcat(Nf, Nr, Delta)
            
        zdot = ca.vertcat(ev['s_dot'], ev['y_dot'], ev['ths_dot'], v1_dot, v2_dot, w3_dot)
        dae = ca.vertcat(Nf - Nf_required, Nr - Nr_required, Delta - Delta_required)
        self.setup_dae_integrator(z, u, g, zdot, dae, param_terms, f_param_terms, pose)
            
        # helper functions
        self.f_N_full = ca.Function('N', [z, u, g, param_terms], [ca.vertcat(Nfr, Nfl, Nrr, Nrl)])
        self.f_N = self.sx2mx_dae('N', [Nfr, Nfl, Nrr, Nrl], z, g, u, param_terms, f_param_terms, pose)
        self.f_y = self.sx2mx_dae('y', [y_fr, y_fl, y_rr, y_rl], z, g, u, param_terms, f_param_terms, pose)
        self.f_sigma = self.sx2mx_dae('sigma', [sigma_fr, sigma_fl, sigma_rr, sigma_rl], z, g, u, param_terms, f_param_terms, pose)
        self.f_alpha = self.sx2mx_dae('alpha', [alpha_fr, alpha_fl, alpha_rr, alpha_rl], z, g, u, param_terms, f_param_terms, pose)
        
        self.throttle_map_a, self.throttle_map_s = self.vehicle_config.tire.get_throttle_map()
        
    def state2zu(self, state: VehicleState):
        z = [state.p.s, state.p.y, state.p.ths, state.v.v1, state.v.v2, state.w.w3]
        if state.u.a != 0:
            s = self.throttle_map_s(state.u.a)
            u = [s, s, s, s, state.u.y]
        else:
            u = [state.tfr.s, state.tfl.s, state.trr.s, state.trl.s, state.u.y]
        return z,u
        
    def state2zug(self, state: VehicleState, g0 = None):
        z, u = self.state2zu(state)
        
        if g0 is None:
            g0 = [self.m * self.g * self.lr / self.L,
                  self.m * self.g * self.lf / self.L,
                  0]
        
        g = self.f_zu2g(z, u, g0)
        g = np.array(g).squeeze().tolist()
        
        return z, u, g
    
    def zn2state(self, state, zn, u): 
        state.tfr.s = u[0].__float__()
        state.tfl.s = u[1].__float__()
        state.trr.s = u[2].__float__()
        state.trl.s = u[3].__float__()
        state.u.y   = u[4].__float__()
        
        state.p.s   = zn[0].__float__()
        state.p.y   = zn[1].__float__()
        state.p.ths = zn[2].__float__() 
        state.v.v1  = zn[3].__float__() 
        state.v.v2  = zn[4].__float__() 
        state.w.w3  = zn[5].__float__() 
        
        state.p.n = self.h if self.vehicle_config.road_surface else 0
        state.u.a = self.throttle_map_a((state.tfr.s + state.tfl.s + state.trr.s + state.trl.s)/4)
        
        self.surf.frenet_to_global(state)
        return
    
    def zngn2state(self, state, zn, gn, u):
        self.zn2state(state, zn, u)
        
        N = self.f_N(zn, gn, u)
        y = self.f_y(zn, gn, u)
        sigma = self.f_sigma(zn, gn, u)
        alpha = self.f_alpha(zn, gn, u)

        state.tfr.y = y[0].__float__()
        state.tfr.a = alpha[0].__float__()
        state.tfr.N = N[0].__float__()
        
        state.tfl.y = y[1].__float__()
        state.tfl.a = alpha[1].__float__()
        state.tfl.N = N[1].__float__()
        
        state.trr.y = y[2].__float__()
        state.trr.a = alpha[2].__float__()
        state.trr.N = N[2].__float__()
        
        state.trl.y = y[3].__float__()
        state.trl.a = alpha[3].__float__()
        state.trl.N = N[3].__float__()
        
        state.fb.f3 = (N[0] + N[1] + N[2] + N[3]).__float__()
        return
    
    def du2state(self, state, du):
        state.du.y = du[4].__float__()
    
    def state2du(self, state):
        return [0, 0, 0, 0, state.du.y]
            
    def zu(self):
        return [self.surf.s_max(),  self.surf.y_max(), np.inf, np.inf, np.inf, np.inf]
        
    def zl(self):
        return [self.surf.s_min(), self.surf.y_min(), -np.inf, -np.inf, -np.inf, -np.inf]
    
    def uu(self):
        return [*[self.vehicle_config.tire.s_max]*4, self.vehicle_config.y_max]
    
    def ul(self):
        return [*[self.vehicle_config.tire.s_min]*4, self.vehicle_config.y_min]
    
    def duu(self):
        return [*[self.vehicle_config.tire.ds_max]*4, self.vehicle_config.dy_max]
    
    def dul(self):
        return [*[self.vehicle_config.tire.ds_min]*4, self.vehicle_config.dy_min]
    
    def _add_model_stage_constraints(self, zkj, ukj, gkj, g, lbg, ubg, param_terms = None):
        if param_terms is None:
            N = self.f_N(zkj, gkj, ukj)[0]
        else:
            N = self.f_N_full(zkj, ukj, gkj, param_terms)
        g += [N / self.vehicle_config.N_max]
        ubg += [0.5] * 4
        lbg += [0.] * 4
        return g, ubg, lbg
    
