''' 
example of a car on a surface where its geometry degrees of freedom
can be adjusted by sliders (tangent contact road model)
'''

import imgui

from vehicle_3d.models.tangent_vehicle_model import KinematicVehicleModel, \
    VehicleModelConfig
from vehicle_3d.surfaces.spline_surface import SplineSurface,\
    SplineSurfaceConfig
from vehicle_3d.visualization.window import Window

surf_config = SplineSurfaceConfig()
test_surf = SplineSurface(surf_config)
model = KinematicVehicleModel(VehicleModelConfig(), test_surf)
test_state = model.get_empty_state()

window = Window(test_surf)
for label, item in model.generate_visual_assets(window.ubo).items():
    window.add_object(label, item)

def _window_extras():
    test_state.p.s = imgui.slider_float('p.s', test_state.p.s,
                            min_value = test_surf.s_min(), max_value = test_surf.s_max())[1]
    test_state.p.y = imgui.slider_float('p.y', test_state.p.y,
                            min_value = test_surf.y_min(), max_value = test_surf.y_max())[1]
    test_state.ths = imgui.slider_float('ths', test_state.ths,
                                        min_value = -2, max_value = 2)[1]
window.draw_vehicle_info = _window_extras
model.update_visual_assets(test_state)
while window.draw():
    model.update_visual_assets(test_state)
    model.zu2state(test_state, *model.state2zu(test_state))
