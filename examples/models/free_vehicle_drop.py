''' example of a car dropping onto a surface '''
import glfw

from vehicle_3d.surfaces.utils import load_surface
from vehicle_3d.visualization.window import Window
from vehicle_3d.models.free_vehicle_model import FreeVehicleModel, \
    FreeVehicleModelConfig


surface = load_surface('hill')
model = FreeVehicleModel(FreeVehicleModelConfig(), surface)

vehicle_state = model.get_empty_state()
vehicle_state.p.s = 15
vehicle_state.p.n = 10
model.zu2state(vehicle_state, *model.state2zu(vehicle_state))

window = Window(surface)
for label, item in model.generate_visual_assets(window.ubo).items():
    window.add_object(label, item)
    model.update_visual_assets(vehicle_state)

print('press enter to start')
while window.step(vehicle_state):
    if window.impl.io.keys_down[glfw.KEY_ENTER]:
        break

while window.step(vehicle_state):
    model.step(vehicle_state)
    model.update_visual_assets(vehicle_state)
