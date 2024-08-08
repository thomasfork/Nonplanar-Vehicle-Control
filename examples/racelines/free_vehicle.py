''' experimental raceline with unconstrained road model vehicle '''
from vehicle_3d.raceline.free_vehicle_raceline import FreeSlipInputRaceline, RacelineConfig, \
    FreeVehicleModelConfig
from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.visualization.raceline_window import RacelineWindow
test_surf = load_surface('tube_turn_closed')
solver = FreeSlipInputRaceline(test_surf, RacelineConfig(), FreeVehicleModelConfig())
RacelineWindow(
    test_surf,
    solver.model,
    solver.solve()
)
