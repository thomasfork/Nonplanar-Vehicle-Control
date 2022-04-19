'''
Attempts to compute basic overtaking maneuvers
'''
from dataclasses import dataclass, field
import time
import numpy as np
import casadi as ca

from barc3d.surfaces import load_surface

from barc3d.utils.collocation_utils import get_collocation_coefficients, interpolate_collocation
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.pytypes import PythonMsg, VehicleConfig

from barc3d.raceline.base_raceline import BaseRaceline, RacelineConfig, RacelineResults

from barc3d.dynamics.dynamics_3d import KinematicBicyclePlanar
from barc3d.raceline.planar_kinematic_bicycle_raceline import PlanarKinematicBicycleRaceline

from barc3d.dynamics.dynamics_3d import KinematicBicycle3D
from barc3d.raceline.kinematic_bicycle_raceline import KinematicBicycleRaceline

from barc3d.dynamics.dynamics_3d import DynamicBicycle3D
from barc3d.raceline.dynamic_bicycle_raceline import DynamicBicycleRaceline

from barc3d.dynamics.dynamics_3d import DynamicTwoTrackSlipInput3D
from barc3d.raceline.dynamic_two_track_slip_input_raceline import DynamicTwoTrackSlipInputRaceline


@dataclass
class EgoConfig(PythonMsg):
    '''configuration for ego agent in overtaking'''
    s0: float = field(default = 0)
    y0: float = field(default = None)
    ths0: float = field(default = 0)
    v0: float = field(default = None)
    y_sep_l: float = field(default = 0.0)
    y_sep_r: float = field(default = 0.0)
    R: np.ndarray = field(default = 0.001) 
    dR: np.ndarray = field(default = 0.001)
    label: str = field(default = 'Ego Vehicle')
    
    agent_class: 'Agent' = None
    
    def __post_init__(self):
        if self.agent_class is None:
            self.agent_class = DynamicTwoTrackAgent
        

@dataclass
class TargetConfig(PythonMsg):
    '''configuration for target agent in overtaking'''
    s0: float = field(default = 10)
    y0: float = field(default = None)
    ths0: float = field(default = 0)
    v0: float = field(default = None)
    y_sep_l: float = field(default = 0.0)
    y_sep_r: float = field(default = 0.0)
    R: np.ndarray = field(default = 0.001)
    dR: np.ndarray = field(default = 0.001)
    label: str = field(default = 'Target Vehicle')

    agent_class: 'Agent' = None
    
    def __post_init__(self):
        if self.agent_class is None:
            self.agent_class = KinematicBicycleAgent

@dataclass
class OvertakingConfig(PythonMsg):
    '''general configuration for overtaking'''
    K: int = field(default = 7)
    N: int = field(default = 30)
    sf: float = field(default = None)
    r: float = field(default = None)

    ego_config: EgoConfig = field(default = None)
    target_config: TargetConfig = field(default = None)
    vehicle_config: VehicleConfig = field(default = None)
    
    verbose: bool = field(default = True)
    
    def __post_init__(self):
        if self.ego_config is None:
            self.ego_config = EgoConfig()
        if self.target_config is None:
            self.target_config = TargetConfig()
        if self.vehicle_config is None:
            self.vehicle_config = VehicleConfig()


class Agent(BaseRaceline):
    '''
    An agent for offline overtaking problems based on raceline optimization
    '''
    def __init__(self, surf: BaseSurface, config: OvertakingConfig, agent_config):
    
        self.agent_config = agent_config
        self.overtaking_config = config
        
        raceline_config = RacelineConfig()
        raceline_config.N = self.overtaking_config.N
        raceline_config.K = self.overtaking_config.K
        raceline_config.closed = False
        raceline_config.R = self.agent_config.R
        raceline_config.dR = self.agent_config.dR
        raceline_config.y_sep_l = self.agent_config.y_sep_l
        raceline_config.y_sep_r = self.agent_config.y_sep_r
        raceline_config.y0 = self.agent_config.y0
        raceline_config.ths0 = self.agent_config.ths0
        raceline_config.v0 = self.agent_config.v0
        raceline_config.v_ws = self.agent_config.v0
        raceline_config.verbose = self.overtaking_config.verbose
        
        super().__init__(surf, config = raceline_config, vehicle_config = self.overtaking_config.vehicle_config, 
                         setup = False)

    def _get_raceline_label(self):
        ''' returns a string label for the raceline '''
        return self.agent_config.label
        
    def _get_raceline_color(self) -> list:
        ''' return a RGBA color for the racleine, ie. [1,0,0,1]'''
        return [1,0,0,1] if isinstance(self.agent_config, EgoConfig) else [0,1,1,1]
        
    def get_stackelberg_interp(self):
        '''
        meant to get an interpolation object from this agent for an agent to use for a stackelberg game
        resulting interpolation object can be used in an optimization problem.
        
        returns interpolation object that takes (t, x), the raceline solution vector 'x', and the raceline
        '''
        self.setup(s0 = self.agent_config.s0, sf = self.overtaking_config.sf)
        target_raceline = self.solve()
        _, w_target, _, _, _, _, _, _, H_target, Z_target, _, _, _ = self._build_opt(s0 = self.agent_config.s0, sf = self.overtaking_config.sf)
        target_z_interp = interpolate_collocation(w_target, H_target, Z_target, self.config)
        return target_z_interp, self.sol['x'], target_raceline
        
    def setup_stackelberg(self, target_agent: 'Agent'):
        t0 = time.time()
        target_z_interp, target_sol, target_raceline = target_agent.get_stackelberg_interp()
        
        self.setup(self.agent_config.s0, self.overtaking_config.sf)
        self.ws_raceline = self.solve()
        
        J, w, w0, ubw, lbw, g, ubg, lbg, H, Z, U, dU, G = self._build_opt(s0 = self.agent_config.s0, sf = self.overtaking_config.sf)
        w0 = self.sol['x']
        
        # add target raceline avoidance constraints
        
        ts = 0 # time at the start of the interval, to interpolate target_z by time
        tau, _, _, _ = get_collocation_coefficients(self.config.K)
        for k in range(self.config.N):
            for j in range(0, self.config.K+1):
                t = ts + tau[j] * H[k] # time at collocation point

                ego_state = Z[k,j]
                target_state = target_z_interp(t, target_sol)

                ds = ego_state[0] - target_state[0]
                dy = ego_state[1] - target_state[1]
                d = ds**2 + dy**2

                g = ca.vertcat(g, d)
                lbg += [self.overtaking_config.r**2*4]
                ubg += [np.inf]
            ts = ts + H[k]
            
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
        
        return target_raceline

class PlanarKinematicBicycleAgent(Agent, PlanarKinematicBicycleRaceline):
    ''' planar kinematic bicycle agent'''
    pass

class KinematicBicycleAgent(Agent, KinematicBicycleRaceline):
    ''' nonplanar kinematic bicycle agent'''
    pass

class DynamicBicycleAgent(Agent, DynamicBicycleRaceline):
    ''' nonplanar dynamic bicycle agent'''
    pass
    
class DynamicTwoTrackAgent(Agent, DynamicTwoTrackSlipInputRaceline):
    ''' nonplanar two track slip input agent'''
    pass


class StackelbergOvertaking():
    '''
    Overtaking based loosely on a Stackelberg Equillibrium
    both agents solve their racelines independently
    the ego agent then resolves using the target raceline as a collosion avoidance constraint
    with its previous raceline as a warm start
    '''
    solve_time = -1
    setup_time = -1
    def __init__(self, surf: BaseSurface, config: OvertakingConfig):
        self.surf = surf
        self.config = config

        if self.config.sf is None:
            self.config.sf = self.surf.s_max()

        if self.config.sf is None:
            self.config.sf = self.surf.s_max()

        if self.config.r is None:
            self.config.r = np.sqrt(self.config.vehicle_config.tf**2 +
                                    self.config.vehicle_config.lf**2) * 1.3

        self.setup()

    def setup(self):
        '''
        solves independent racelines and sets up Stackelberg Equillibrium problem
        '''
        t0 = time.time()
        
        ego_class = self.config.ego_config.agent_class
        target_class = self.config.target_config.agent_class
        
        ego_agent    = ego_class(self.surf, self.config, self.config.ego_config)
        target_agent = target_class(self.surf, self.config, self.config.target_config)
        target_raceline = ego_agent.setup_stackelberg(target_agent)

        self.ego_agent = ego_agent
        self.target_agent = target_agent
        self.target_raceline = target_raceline
        self.setup_time = time.time() - t0

    def solve(self):
        '''
        Solves Stackelberg problem once set up
        '''
        ego_raceline = self.ego_agent.solve()

        self.ego_agent.plot_raceline_3D([self.target_raceline, ego_raceline], circular_buf = self.config.r, closed = False)


