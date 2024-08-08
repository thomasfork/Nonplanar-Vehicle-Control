''' example of a two track raceline '''

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.raceline.base_raceline import RacelineConfig
from vehicle_3d.raceline.two_track_slip_input_raceline import SlipInputRaceline, \
    VehicleModelConfig
from vehicle_3d.visualization.raceline_window import RacelineWindow

surf = load_surface('tube_bend')
solver = SlipInputRaceline(
    surf,
    RacelineConfig(),
    VehicleModelConfig()
)

RacelineWindow(
    solver.surf,
    solver.model,
    solver.solve()
)
