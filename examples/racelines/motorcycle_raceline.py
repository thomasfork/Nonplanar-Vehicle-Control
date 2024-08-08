''' example of a motorcycle raceline '''

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.raceline.base_raceline import RacelineConfig
from vehicle_3d.raceline.motorcycle_raceline import MotorcycleConfig, MotorcycleRaceline
from vehicle_3d.visualization.raceline_window import RacelineWindow

surf = load_surface('pump_track')
solver = MotorcycleRaceline(
    surf,
    RacelineConfig(N=50),
    MotorcycleConfig()
)

RacelineWindow(
    solver.surf,
    solver.model,
    solver.solve(),
    instance_interval=.1, # increase this to ~ 1 second if viewing on smaller GPU's
    fullscreen=False,
)
