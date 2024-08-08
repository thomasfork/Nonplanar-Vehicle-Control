''' 
example of a car on a surface where its geometry degrees of freedom
can be adjusted by sliders (unconstrained road model)
'''

import imgui

from vehicle_3d.models.free_vehicle_model import FreeSlipInputVehicleModel, \
    FreeVehicleModelConfig
from vehicle_3d.surfaces.spline_surface import SplineSurface,\
    SplineSurfaceConfig
from vehicle_3d.visualization.window import Window

surf_config = SplineSurfaceConfig()
test_surf = SplineSurface(surf_config)
model = FreeSlipInputVehicleModel(FreeVehicleModelConfig(), test_surf)
test_state = model.get_empty_state()

window = Window(test_surf)
for label, item in model.generate_visual_assets(window.ubo).items():
    window.add_object(label, item)

def _window_extras():
    test_state.p.s = imgui.slider_float('p.s', test_state.p.s,
                            min_value = test_surf.s_min(), max_value = test_surf.s_max())[1]
    test_state.p.y = imgui.slider_float('p.y', test_state.p.y,
                            min_value = test_surf.y_min(), max_value = test_surf.y_max())[1]
    test_state.p.n = imgui.slider_float('p.n', test_state.p.n,
                            min_value = 0, max_value = 5)[1]
    test_state.r.a = imgui.slider_float('r.a', test_state.r.a,
                                        min_value = -2, max_value = 2)[1]
    test_state.r.b = imgui.slider_float('r.b', test_state.r.b,
                                        min_value = -2, max_value = 2)[1]
    test_state.r.c = imgui.slider_float('r.c', test_state.r.c,
                                        min_value = -2, max_value = 2)[1]
    test_state.u.y = imgui.slider_float('u.y', test_state.u.y,
                                        min_value = -2, max_value = 2)[1]
window.draw_vehicle_info = _window_extras
model.update_visual_assets(test_state)
while window.draw():
    model.update_visual_assets(test_state)
    model.zu2state(test_state, *model.state2zu(test_state))
