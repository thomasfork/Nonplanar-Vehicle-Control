'''
example of a motorcycle on a surface where its geometry degrees of freedom
can be adjusted by sliders 
'''
import imgui

from vehicle_3d.models.motorcycle_model import MotorcycleDynamics, MotorcycleConfig
from vehicle_3d.surfaces.spline_surface import SplineSurface,\
    SplineSurfaceConfig
from vehicle_3d.visualization.window import Window

surf_config = SplineSurfaceConfig()
test_surf = SplineSurface(surf_config)
model = MotorcycleDynamics(MotorcycleConfig(), test_surf)
test_state = model.get_empty_state()

window = Window(test_surf)
window.imgui_width += 150
for label, item in model.generate_visual_assets(window.ubo).items():
    window.add_object(label, item)

def _window_extras():
    test_state.p.s = imgui.slider_float('p.s', test_state.p.s,
                            min_value = test_surf.s_min(), max_value = test_surf.s_max())[1]
    test_state.p.y = imgui.slider_float('p.y', test_state.p.y,
                            min_value = test_surf.y_min(), max_value = test_surf.y_max())[1]
    test_state.ths = imgui.slider_float('ths', test_state.ths,
                                        min_value = -2, max_value = 2)[1]
    test_state.c = imgui.slider_float('c', test_state.c,
                                        min_value = -2, max_value = 2)[1]
    test_state.d = imgui.slider_float('d', test_state.d,
                                        min_value = -2, max_value = 2)[1]
    test_state.u.y = imgui.slider_float('u.y', test_state.u.y,
                                        min_value = -2, max_value = 2)[1]
    _ = imgui.slider_float('front camber', test_state.tf.c,
                                        min_value = -2, max_value = 2)[1]
    _ = imgui.slider_float('front steering', test_state.tf.y,
                                        min_value = -2, max_value = 2)[1]
window.draw_vehicle_info = _window_extras
model.update_visual_assets(test_state)
while window.draw():
    model.update_visual_assets(test_state)
    model.zua2state(test_state, *model.state2zua(test_state))
