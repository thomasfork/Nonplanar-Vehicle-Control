'''
code for 3D plotting of one or more racelines
this is an offline visualizer, the racelines are not supposed to change,
just be looped over and over
includes tools to start/stop the animation,
reverse the animation,
manually drag through the animation,
or reverse the animation player

all racelines must have the same configuration for N, K or errors will occur.
'''
# pylint: disable=line-too-long
import time
from typing import Dict, List, Tuple, Union
from dataclasses import asdict

import numpy as np
import imgui

from vehicle_3d.pytypes import BaseBodyState
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.dynamics_model import DynamicsModel, DAEDynamicsModel
from vehicle_3d.utils.ocp_util import OCPResults
from vehicle_3d.obstacles.polytopes import RectangleObstacle
from vehicle_3d.visualization.utils import get_cmap_rgba, plot_multiline, \
    IMGUI_WHITE, IMGUI_BLACK
from vehicle_3d.visualization.objects import gl, VertexObject
from vehicle_3d.visualization.window import Window, camera_follow_view_mat

NUMBER_OF_TRACES = 4

class RacelineWindow(Window):
    '''
    the raceline window class
    '''
    dom: BaseSurface

    selected_results_index: int = 0
    running: bool = True
    reverse: bool = False
    animation_t:float = 0
    animation_dt: float = 0
    animation_t0:float = -1
    animation_t_scale:float = 1

    camera_follow: bool = True

    t: np.ndarray
    available_plot_vars: Dict[str, List[np.ndarray]]
    available_abscissa_labels: List[str]
    selected_abscissa_vars: List[int]
    available_plot_labels: List[str]
    selected_plot_vars: List[int]

    raceline_objects: Dict[str, Tuple[bool, Dict[str, Tuple[bool, VertexObject]]]]
    selected_raceline_index: int = 0

    def __init__(self,
            surf: BaseSurface,
            models: List[Union[DynamicsModel, DAEDynamicsModel]],
            results: List[OCPResults],
            obstacles: List[RectangleObstacle] = None,
            fullscreen:bool = False,
            skybox:bool = True,
            run:bool = True,
            instance_interval: float = 0.33):
        if not isinstance(models, list):
            models = [models]
        if not isinstance(results, list):
            results = [results]

        self.models = models
        self.model_states = [model.get_empty_state() for model in models]
        self.results = results

        self.raceline_objects = {}
        self._ensure_unique_result_labels()

        super().__init__(surf, obstacles, fullscreen, skybox)
        self._precompute_parameters()
        self._generate_raceline_objects(instance_interval)

        if run:
            self.run()

    def add_raceline_object(self, result_label: str, name, obj, show = True):
        ''' similar to add_object but creates a reference in self.raceline_objects '''
        if result_label not in self.raceline_objects:
            self.raceline_objects[result_label] = [True, {}]
        self.raceline_objects[result_label][1][name] = [show, obj]

    def run(self):
        ''' run the visualization loop until the window is closed'''
        while not self.should_close:
            self.draw()
        self.close()

    def step(self, state: BaseBodyState):
        raise NotImplementedError('Raceline figure does not support online updates')

    def draw(self):
        self._update_animation_time()
        self._update_vehicles()
        super().draw()

    def _ensure_unique_result_labels(self):
        ''' make sure all results have unique labels'''
        used_labels = []
        for result in self.results:
            original_label = result.label
            k = 1
            while result.label in used_labels:
                result.label = original_label + f' ({k})'
                k += 1
            used_labels.append(result.label)

    def _precompute_parameters(self):

        self.available_plot_vars = {}
        # default fields to show first
        self.available_plot_vars['t'] = []
        self.available_plot_vars['v'] = []
        self.available_plot_vars['p.s'] = []
        self.available_plot_vars['p.y'] = []
        self.available_plot_vars['ths'] = []

        self.available_plot_vars['v'] = [np.array([state.vb.signed_mag()
            for state in result.states]) for result in self.results]

        def _add_plot_entries(state_dict: Dict[str, Union[dict, float]], label_prefix = ''):
            for label, item in state_dict.items():
                if isinstance(item, dict):
                    _add_plot_entries(
                        state_dict[label],
                        label_prefix = label_prefix + label + '.')
                else:
                    full_label = label_prefix + label
                    if full_label not in self.available_plot_vars:
                        self.available_plot_vars[full_label] = []

        for result in self.results:
            _add_plot_entries(asdict(result.states[0]))

        def _entry_data_getter(label:str):
            attrs = label.split('.')
            def getter(item):
                for attr in attrs:
                    item = getattr(item, attr)
                return item
            return getter

        def _populate_plot_entries(items: List[BaseBodyState]):
            for label, plot_items in self.available_plot_vars.items():
                if label == 'v':
                    continue
                getter = _entry_data_getter(label)
                try:
                    entry = np.array([getter(state) for state in items])

                    plot_items.append(entry)
                except AttributeError:
                    plot_items.append(None)

        for result in self.results:
            _populate_plot_entries(result.states)
        self.t = self.available_plot_vars['t']

        self.available_plot_labels = list(self.available_plot_vars.keys())
        self.selected_plot_vars = list(range(1, 1+NUMBER_OF_TRACES))

        self.available_abscissa_labels = list(self.available_plot_vars.keys())
        self.selected_abscissa_vars = [0] * NUMBER_OF_TRACES

        self.colors = [imgui.get_color_u32_rgba(*result.color) for result in self.results]
        self.labels = [result.label for result in self.results]
        self.show_raceline = [True] * len(self.labels)
        self.times  = [result.time for result in self.results]

        self.t_min = min(tk.min() for tk in self.t)
        self.t_max = max(tk.max() for tk in self.t)
        self.v_min = min(vk.min() for vk in self.available_plot_vars['v'])
        self.v_max = max(vk.max() for vk in self.available_plot_vars['v'])

    def _generate_raceline_objects(self, instance_interval:float = 1.0):
        for result, model in zip(self.results, self.models):

            racers = result.triangulate_instanced_trajectory(
                self.ubo,
                np.linspace(0, result.time, int(result.time/instance_interval)),
                model,
                self.dom
            )
            self.add_raceline_object(result.label, 'Racers', racers, show=False)

            raceline = result.triangulate_trajectory(
                self.ubo,
                model,
                self.dom,
                v_max = self.v_max,
                v_min = self.v_min,
                n = min(1000, self.dom.triangulate_num_s())
            )
            self.add_raceline_object(result.label, 'Raceline', raceline)

            for asset_label, asset in model.generate_visual_assets(self.ubo).items():
                self.add_raceline_object(result.label, asset_label, asset)

    def _draw_opengl(self):
        # pylint: disable=unsupported-binary-operation
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        for _, (show, objects) in self.raceline_objects.items():
            if show:
                for _, (draw_obj, obj) in objects.items():
                    if draw_obj:
                        obj.draw()
        for _, (draw_obj, obj) in self.window_objects.items():
            if draw_obj:
                obj.draw()

        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        for _, (draw_obj, obj) in self.translucent_window_objects.items():
            if draw_obj:
                obj.draw()
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glDepthMask(gl.GL_TRUE)

    def draw_vehicle_info(self):
        self._draw_comparison()

    def draw_extras(self):
        self._draw_colorbar()

    def _draw_colorbar(self):
        w = 50
        w_tot = w + 55
        top_pad = 50
        left_pad = 30
        h = self.window_height - top_pad*2

        draw_list = imgui.get_overlay_draw_list()
        C_array = get_cmap_rgba(np.flip(np.arange(h)), 0, h)

        for (C,py) in zip(C_array, range(top_pad, top_pad+h)):
            color = imgui.get_color_u32_rgba(*C)
            draw_list.add_line(left_pad, py, left_pad + w, py, color, thickness=1)

        n = 8

        draw_list.add_rect_filled(left_pad, top_pad-16, left_pad + 160, top_pad, IMGUI_WHITE)
        draw_list.add_text(left_pad +3, top_pad-16, IMGUI_BLACK, 'Vehicle Speed (m/s)')
        for v, py in zip(np.linspace(self.v_max, self.v_min, n), np.linspace(top_pad, top_pad+h, n)):
            draw_list.add_line(left_pad, py, left_pad + w_tot, py, IMGUI_BLACK, thickness=1)
            y_offset = -16 if v == self.v_min else 3
            draw_list.add_rect_filled(left_pad + w +3, py + y_offset, left_pad + w_tot, py + y_offset + 16, IMGUI_WHITE)
            draw_list.add_text(left_pad + w +3, py + y_offset, IMGUI_BLACK, f'{v:6.2f}')

    def _update_animation_time(self):
        if self.running:
            if self.animation_t0 > 0 :
                dt = (time.time() - self.animation_t0) * self.animation_t_scale
            else:
                dt = 0
            self.animation_t += (-dt if self.reverse else dt)
            self.animation_dt = (-dt if self.reverse else dt)

            if self.dom.periodic:
                if self.animation_t < self.t_min:
                    self.animation_t = self.t_max
                if self.animation_t > self.t_max:
                    self.animation_t = self.t_min
            else:
                if self.animation_t < self.t_min - 2:
                    self.animation_t = self.t_max + 2
                if self.animation_t > self.t_max + 2:
                    self.animation_t = self.t_min - 2
        else:
            self.animation_dt = 0.
        self.animation_t0 = time.time()

    def _draw_comparison(self):
        imgui.columns(2)
        if imgui.radio_button("Play", self.running):
            self.running = not self.running
        imgui.next_column()
        if imgui.radio_button("Reverse", self.reverse):
            self.reverse = not self.reverse

        imgui.columns(1)

        _, self.animation_t  = imgui.slider_float('Time',self.animation_t, min_value = self.t_min, max_value = self.t_max)
        _, self.animation_t_scale = imgui.slider_float('Speed',self.animation_t_scale, min_value = 0.1, max_value = 10,flags=imgui.SLIDER_FLAGS_LOGARITHMIC)

        # draw legend
        dy = 20
        draw_list = imgui.get_window_draw_list()

        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()

        px = org.x + off.x
        py = org.y + off.y

        for color, label, t in zip(self.colors, self.labels, self.times):
            draw_list.add_rect_filled(px, py+3 , px+30, py+15, color)
            if t is not None and label is not None:
                draw_list.add_text(px+35, py, IMGUI_WHITE, f'{label} ({t:0.2f}s)')
            else:
                draw_list.add_text(px+35, py, IMGUI_WHITE, f'{label}')
            py += dy

        imgui.set_cursor_screen_pos((px, py + dy))

        # draw time series info on raceline(s)
        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()
        y0 = org.y + off.y
        plot_size = (0, int((self.window_height - y0)/NUMBER_OF_TRACES) - 52)

        generic_args = {
                'colors': self.colors,
                'labels': self.labels,
                'size': plot_size,
                'show_title': False
            }

        for k in range(NUMBER_OF_TRACES):
            imgui.columns(2)

            _, self.selected_abscissa_vars[k] = \
                imgui.combo(f'##trace_x {k+1}', self.selected_abscissa_vars[k], self.available_abscissa_labels)
            imgui.next_column()
            _, self.selected_plot_vars[k] = \
                imgui.combo(f'##trace {k+1}', self.selected_plot_vars[k], self.available_plot_labels)
            imgui.columns(1)

            abs_label = self.available_abscissa_labels[self.selected_abscissa_vars[k]]
            var_label = self.available_plot_labels[self.selected_plot_vars[k]]
            show_tooltip = abs_label in ['t', 'p.s']
            tooltip_label = 's' if abs_label == 'p.s' else 't'

            x = self.available_plot_vars[abs_label]
            y = self.available_plot_vars[var_label]

            current_x = [np.interp(self.animation_t, t, data) if data is not None else None for t, data in zip(self.t, x)]
            current_y = [np.interp(self.animation_t, t, data) if data is not None else None for t, data in zip(self.t, y)]

            plot_multiline(
                time = x,
                data = y,
                current_time=current_x,
                current_data=current_y,
                title = f'##trace plot {k}',
                show_tooltip = show_tooltip,
                tooltip_label = tooltip_label,
                **generic_args)

    def _draw_camera_menu(self):
        imgui.set_next_window_position(self.window_width - self.imgui_width - 200, 0)
        imgui.set_next_window_size(200, 0) # autosize y
        expanded, _ = imgui.begin("Camera Settings",
            closable = False)
        if expanded:
            if imgui.button("Reset Camera"):
                self._reset_camera()
            if imgui.radio_button('Camera Follow', self.camera_follow):
                self.camera_follow = not self.camera_follow
            if self.camera_follow:
                imgui.push_item_width(-1)
                if len(self.results) > 1:
                    _, self.selected_raceline_index = imgui.combo(
                        "##Selected Raceline", self.selected_raceline_index, self.labels)

                _, self._camera_follow_mode = imgui.combo(
                    "##Selected Follow Mode",
                    self._camera_follow_mode,
                    self._camera_follow_mode_labels)
                imgui.pop_item_width()

            imgui.separator()
            imgui.text('Hide / Show:')

            for label, (show, objects) in self.raceline_objects.items():
                if not show:
                    if imgui.radio_button('     ' + label, show):
                        show = not show
                        self.raceline_objects[label][0] = show

                else:
                    imgui.columns(2)
                    imgui.set_column_width(-1, 30)
                    if imgui.radio_button('##' + label, show):
                        show = not show
                        self.raceline_objects[label][0] = show

                    imgui.next_column()
                    expanded, _ = imgui.collapsing_header(label + '##header')
                    if expanded:
                        for name, (draw_obj, _) in objects.items():
                            if imgui.radio_button(name + '##' + label, draw_obj):
                                self.raceline_objects[label][1][name][0] = not draw_obj

                    imgui.columns(1)

            for label, (draw_obj, _) in self.window_objects.items():
                if imgui.radio_button(label, draw_obj):
                    self.window_objects[label][0] = not draw_obj
            for obj_name, (draw_obj, _) in self.translucent_window_objects.items():
                if imgui.radio_button(obj_name, draw_obj):
                    self.translucent_window_objects[obj_name][0] = not draw_obj

            imgui.separator()
            imgui.text('Ground Plane Offset:')
            changed, self.ground_plane_height  = imgui.input_float(
                '##Ground Plane Height',
                self.ground_plane_height)
            if changed:
                mat = np.eye(4,dtype=np.float32)
                mat[2,3] = self.ground_plane_height
                self.window_objects['Ground Plane'][1].update_pose(mat = mat)

        imgui.end()

    def _update_vehicles(self):
        for result, model, state in zip(self.results, self.models, self.model_states):
            t = np.clip(self.animation_t, 0, result.time)
            z = result.z_interp(t)
            u = result.u_interp(t)

            if not isinstance(model, DAEDynamicsModel):
                model.zu2state(state, z, u)
            else:
                a = result.a_interp(t)
                model.zua2state(state, z, u ,a)
            model.update_visual_assets(state, self.animation_dt)

            if self.camera_follow:
                if result.label == self.labels[self.selected_raceline_index]:
                    self.update_camera_pose(
                        camera_follow_view_mat(
                            self._camera_follow_modes[self._camera_follow_mode],
                            state,
                            self.dom)
                    )
