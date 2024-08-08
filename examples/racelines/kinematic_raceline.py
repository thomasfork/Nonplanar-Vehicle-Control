''' example of a kinematic bicycle raceline '''

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.raceline.base_raceline import RacelineConfig
from vehicle_3d.raceline.kinematic_bicycle_raceline import KinematicRaceline,\
    VehicleModelConfig
from vehicle_3d.visualization.raceline_window import RacelineWindow

surf = load_surface('tube_bend')
solver = KinematicRaceline(
    surf,
    RacelineConfig(),
    VehicleModelConfig()
)

RacelineWindow(
    solver.surf,
    solver.model,
    solver.solve()
)
