'''
template for any model, not necessarily dynamic
establishes standard i/o
'''
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Tuple

import casadi as ca
import numpy as np

from vehicle_3d.pytypes import PythonMsg

@dataclass
class ModelConfig(PythonMsg):
    ''' model config template '''

@dataclass
class ModelVars(PythonMsg):
    ''' model variable template '''
    u: ca.SX = field(default = None)
    ''' inputs to the model'''
    g: ca.SX = field(default = None)
    ''' outputs from the model '''
    ubg: List[float] = field(default = None)
    ''' upper bound on outputs '''
    lbg: List[float] = field(default = None)
    ''' lower bound on outputs '''
    input_dim: float = field(default = None)
    ''' dimension of input vector '''

@dataclass
class StatefulModelVars(ModelVars):
    ''' model variable template with differential state '''
    z: ca.SX = field(default = None)
    ''' differential state of the model'''
    z_dot: ca.SX = field(default = None)
    ''' differential state derivative of the model'''
    state_dim: float = field(default = None)
    ''' dimension of state vector '''

    def get_all_indep_vars(self):
        ''' return a list of state and input vectors (and algebraic state for DAE models )'''
        return [self.z, self.u]


@dataclass
class DAEModelVars(StatefulModelVars):
    ''' model variable template with algebraic state'''
    a: ca.SX = field(default = None)
    ''' algebraic variables '''
    h: ca.SX = field(default = None)
    ''' constraint equation for algebraic variables '''
    alg_dim: float = field(default = None)
    ''' dimension of algebraic state vector '''
    dae_dim: float = field(default = None)
    ''' dimension of algebraic constraint vector '''

    def get_all_indep_vars(self):
        return [self.z, self.u, self.a]

class Model(ABC):
    ''' model class template '''
    config: ModelConfig
    ''' configuration of the model '''
    model_vars: ModelVars
    ''' symbolic attributes of the model '''

class StatefulModel(Model):
    ''' model class template with internal state '''
    model_vars: StatefulModelVars

    @abstractmethod
    def get_empty_state(self) -> PythonMsg:
        ''' obtain an empty state compatible with the model'''

    @abstractmethod
    def state2u(self, state: PythonMsg) -> np.ndarray:
        ''' get input vector of a state for given model '''

    @abstractmethod
    def state2z(self, state: PythonMsg) -> np.ndarray:
        ''' get input vector of a state for given model '''

    def state2zu(self, state: PythonMsg) -> Tuple[np.ndarray, np.ndarray]:
        ''' get state and input vector of a state for given model '''
        return self.state2z(state), self.state2u(state)

    @abstractmethod
    def u2state(self, state: PythonMsg, u: np.ndarray) -> None:
        ''' update input from vector'''

    @abstractmethod
    def zu2state(self, state: PythonMsg, z: np.ndarray, u: np.ndarray) -> None:
        ''' update state and input from vectors, not supported if DAE '''

    @abstractmethod
    def zu(self, s: float = 0) -> List[float]:
        ''' upper bound on state vector '''

    @abstractmethod
    def zl(self, s: float = 0) -> List[float]:
        ''' lower bound on state vector '''

    @abstractmethod
    def uu(self) -> List[float]:
        ''' upper bound on input vector '''

    @abstractmethod
    def ul(self) -> List[float]:
        ''' lower bound on input vector '''

    @abstractmethod
    def duu(self) -> List[float]:
        ''' upper bound on input rate vector '''

    @abstractmethod
    def dul(self) -> List[float]:
        ''' lower bound on input rate vector '''


class StatefulDAEModel(StatefulModel):
    ''' model class template with internal state and DAE constraints '''
    model_vars: DAEModelVars

    @abstractmethod
    def state2a(self, state: PythonMsg) -> np.ndarray:
        ''' get algebraic state vector for given model '''

    def state2zua(self, state:PythonMsg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ''' get state, input and alg. state vectors of a state for given model '''
        return self.state2z(state), self.state2u(state), self.state2a(state)

    def zu2state(self, state, z, u):
        raise NotImplementedError('DAE Requires algebraic state vector as well')

    @abstractmethod
    def zua2state(self, state: PythonMsg, z: np.ndarray, u: np.ndarray, a: np.ndarray) -> None:
        ''' update state and input from vectors'''

    @abstractmethod
    def au(self) -> List[float]:
        ''' upper bound on algebraic state vector '''

    @abstractmethod
    def al(self) -> List[float]:
        ''' lower bound on algebraic state vector '''
