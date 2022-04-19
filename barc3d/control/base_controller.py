from abc import abstractmethod, ABC
from barc3d.pytypes import VehicleState


class BaseController(ABC):
    
    
    @abstractmethod
    def step(self, state:VehicleState):
        '''
        general-purpose single-vehicle method to update control input to vehicle
        controller should map current vehicle state (pytypes.VehicleState) to an control input
        state.u 
        '''
        return
