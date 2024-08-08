'''
Standard methods and features for nonplanar raceline computation and manipulation
'''
from dataclasses import dataclass, field
from abc import abstractmethod
from typing import Optional

import numpy as np

from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.dynamics_model import DynamicsModel, DynamicsModelConfig
from vehicle_3d.utils.ocp_util import OCP, OCPConfig, Fix, COLLOCATION_METHODS, OCPResults

@dataclass
class RacelineConfig(OCPConfig):
    ''' configuration for a raceline solver '''
    verbose: bool = field(default = True)
    closed: Optional[bool] = field(default = None)
    R: np.ndarray  = field(default = 1e-7)
    ''' quadratic input cost '''
    dR: np.ndarray = field(default = 1e-7)
    ''' quadratic input rate cost '''

    v0: Optional[float] = field(default = None)
    ''' optional fixed initial speed for aperiodic racelines '''
    y0: Optional[float] = field(default = None)
    ''' optional fixed initial y coordinate for aperiodic racelines '''
    ths0: Optional[float] = field(default = None)
    ''' optional fixed initial ths coordinate for aperiodic racelines '''


class BaseRaceline(OCP):
    ''' base raceline class '''
    config: RacelineConfig
    def __init__(self,
            surf: BaseSurface,
            config: RacelineConfig,
            model_config: DynamicsModelConfig,
            ws_raceline: OCPResults = None,
            ws_model: DynamicsModel = None):
        config.fix = Fix.S
        if config.closed is None:
            config.closed = surf.config.closed
        elif config.closed and not surf.config.closed:
            config.closed = False
        assert config.method in COLLOCATION_METHODS
        super().__init__(surf, config, model_config, ws_raceline, ws_model, setup=True)

    @abstractmethod
    def _get_state_bounds(self,n,k):
        '''
        modify some limits for forwards-only motion in racelines discretized in space
        these slow down IPOPT iterations but dramatically improve robustness
        especially by avoiding iterations that involve the vehicle moving backwards
        '''
        return super()._get_state_bounds(n,k)

    @abstractmethod
    def _enforce_initial_constraints(self):
        ''' initial constraints for aperiodic racelines '''

    @abstractmethod
    def _guess_z(self,n,k):
        '''
        modify some state guesses for racelines
        '''
        return super()._guess_z(n,k)

    def _unpack_soln(self, sol, solved: bool = False):
        soln = super()._unpack_soln(sol, solved)
        soln.periodic = self.surf.config.closed
        return soln

class BasePlanarRaceline(BaseRaceline):
    ''' base raceline class that forces models to be planar '''

    def __init__(self,
            surf: BaseSurface,
            config: RacelineConfig,
            model_config: DynamicsModelConfig,
            ws_raceline: OCPResults = None,
            ws_model: DynamicsModel = None):
        model_config.build_planar_model = True
        super().__init__(
            surf = surf,
            config = config,
            model_config = model_config,
            ws_raceline = ws_raceline,
            ws_model = ws_model
        )
