from abc import abstractmethod
from barc3d.pytypes import VehicleState, VehicleConfig
from barc3d.surfaces.base_surface import BaseSurface

class BaseFigure():

    @abstractmethod
    def __init__(self, surf: BaseSurface, config: VehicleConfig):
        return
        
    @abstractmethod
    def draw(self, state:VehicleState) -> bool:
        '''
        return True if draw was sucessful, False otherwise
        '''
        return
        
    @abstractmethod
    def close(self) -> None:
        return
   
    @abstractmethod
    def available(self) -> bool:  # backend check
        return False
    
    @abstractmethod 
    def ready(self) -> bool:      # check if figure set up (useful if multithreaded)
        return False
