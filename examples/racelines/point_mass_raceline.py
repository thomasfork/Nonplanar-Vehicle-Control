''' example of a point mass raceline '''

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.raceline.base_raceline import RacelineConfig
from vehicle_3d.raceline.point_raceline import PointRaceline, PointModelConfig
from vehicle_3d.visualization.raceline_window import RacelineWindow

surf = load_surface('tube_bend')
solver = PointRaceline(
    surf,
    RacelineConfig(),
    PointModelConfig()
)

RacelineWindow(
    solver.surf,
    solver.model,
    solver.solve()
)
