'''
base classes for controllers
'''
from abc import abstractmethod, ABC
from vehicle_3d.pytypes import BaseBodyState


class Controller(ABC):
    ''' base class for all controllers'''
    @abstractmethod
    def step(self, state:BaseBodyState) -> None:
        '''
        general-purpose single-vehicle method to update control input to vehicle
        controller should map current vehicle state (pytypes.VehicleState) to an control input
        state.u
        '''

    def get_prediction(self):
        ''' return the latest prediction of the predictive controller '''
        return None


class PredictiveController(Controller):
    '''
    base class for predictive controllers
    predictive controllers should have some way to provide
    their planned trajectory to higher level code
    '''
    solve_time: float = -1
    @abstractmethod
    def get_prediction(self):
        pass
