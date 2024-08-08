'''
plotting / rendering based on OpenGL and Imgui
'''
from typing import Dict, Tuple, List, Callable, Optional, Union
import platform
import time
import ctypes
import os
from enum import Enum
from dataclasses import asdict

import numpy as np
try:
    import cv2
    RECORDING_SUPPORTED = True
except ImportError:
    RECORDING_SUPPORTED = False

import imgui
import glfw
from imgui.integrations.glfw import GlfwRenderer

from vehicle_3d.pytypes import Position, OrientationQuaternion, BodyAngularVelocity, \
    BaseBodyState, Renderable, Domain
from vehicle_3d.utils.load_utils import get_assets_file, get_project_folder
from vehicle_3d.obstacles.polytopes import RectangleObstacle
from vehicle_3d.visualization import glm
from vehicle_3d.visualization.utils import join_vis, plot_multiline, IMGUI_TEAL
from vehicle_3d.visualization.objects import gl, UBOObject, Skybox, GroundPlane, \
    VertexObject

TRACE_LEN = 200
NUMBER_OF_TRACES = 6

class CameraFollowMode(Enum):
    ''' modes for camera follow '''
    VEHICLE = 'Vehicle View'
    VELOCITY = 'Velocity View'
    SURFACE = 'Surface View'
    GLOBAL = 'Global View'

def camera_follow_view_mat(
        mode: CameraFollowMode,
        state: BaseBodyState,
        dom: Domain):
    ''' view transform for a camera follow mode '''
    u_view = np.eye(4)
    if mode == CameraFollowMode.VEHICLE:
        u_view[:3,-1] = -state.q.Rinv() @ (state.x.to_vec())
        u_view[:3,:3] = state.q.Rinv()
    elif mode == CameraFollowMode.VELOCITY:
        vg = state.q.R() @ state.vb.to_vec()
        th = np.arctan2(vg[1], vg[0])
        u_view[:2,:2] = np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])
        u_view[:3,-1] = -u_view[:3,:3] @ (state.x.to_vec())
    elif mode == CameraFollowMode.SURFACE:
        if dom is not None:
            R = dom.camera_follow_mat(state.p.s, state.p.y)
            u_view[:3,:3] = R
            u_view[:3,-1] = -R @ (state.x.to_vec())
    elif mode == CameraFollowMode.GLOBAL:
        u_view[:3,-1] = -state.x.to_vec()
    else:
        raise NotImplementedError(f'Unknown camera follow mode {mode}')
    return u_view

def get_font_file(name = 'DejaVuSans.ttf'):
    ''' helper function to get a path to a provided font '''
    filename = get_assets_file(name)
    return filename


class Window():
    '''
    class for creating a window using GLFW,
    adding objects to it to draw with OpenGL,
    and drawing a GUI using imgui

    Keyboard shortcuts:
      ESC: request a window close (Window.draw() returns false)
      SPACE: toggle showing the GUI
      R: toggle screen recording
    '''
    window = None
    window_open:bool = True
    window_height:int = 1080
    window_width:int = 1920
    show_imgui:bool = True
    _last_show_imgui_change: float = None # debounce toggling show via keyboard (space)
    imgui_width:int = 300
    impl = None
    should_close:bool = False

    ubo: UBOObject
    skybox: Skybox
    _window_read_pbo: int = None
    _recording_window: bool = False
    _last_record_window_change: float = None
    _recording_writer: cv2.VideoWriter

    # camera follow variables
    camera_follow:bool = False
    _camera_follow_mode: int
    _camera_follow_modes: List[CameraFollowMode]
    _camera_follow_mode_labels: List[str]

    # mouse drag variables
    _mouse_drag_prev_delta: Tuple[int, int] = (0,0)
    _drag_mice:List[float] = [0,1,2]
    _drag_mouse:int = -1
    _drag_mouse_callbacks: List[Callable] = None

    window_objects: Dict[str, Tuple[bool, Renderable]]
    translucent_window_objects: Dict[str, Tuple[bool, Renderable]]

    # misc interal states
    ground_plane_height: float = 0

    # state history trace entries
    available_plot_getters: Optional[Dict[str, Callable[[BaseBodyState], float]]] = None
    available_plot_vars: Optional[Dict[str, np.ndarray]] = None
    available_abscissa_labels: List[str]
    selected_abscissa_vars: List[int]
    available_plot_labels: List[str]
    selected_plot_vars: List[int]

    def __init__(self,
            dom: Domain = None,
            obstacles: List[RectangleObstacle] = None,
            fullscreen = False,
            skybox = True,
            ground_plane=False):
        self.dom = dom
        self._fullscreen = fullscreen
        self.obstacles = obstacles

        self._camera_follow_modes = [e for e in CameraFollowMode]
        self._camera_follow_mode_labels = [e.value for e in CameraFollowMode]
        self._camera_follow_mode  = 0

        self.window_objects = {}
        self.translucent_window_objects = {}
        self._drag_mouse_callbacks = {
            0:self._update_left_mouse_drag,
            1:self._update_right_mouse_drag,
            2:self._update_scroll_mouse_drag
        }

        self._create_imgui()
        self._create_window()
        self._create_imgui_renderer()
        self.ubo = UBOObject()
        self._generate_default_objects(skybox, ground_plane)
        self._reset_camera()
        self.update_projection()

    def step(self, state: BaseBodyState) -> bool:
        ''' update camera follow for a state and redraw the window '''
        if self.camera_follow:
            self.update_camera_follow(state)
        if self.available_plot_getters is None:
            self._populate_plottable_vars(state)

        for label, getter in self.available_plot_getters.items():
            self.available_plot_vars[label] = np.roll(self.available_plot_vars[label], 1)
            self.available_plot_vars[label][0] = getter(state)

        return self.draw()

    def _populate_plottable_vars(self, state: BaseBodyState):

        def _entry_data_getter(label:str):
            attrs = label.split('.')
            def getter(item):
                for attr in attrs:
                    item = getattr(item, attr)
                return item
            return getter

        def _add_plot_entries(state_dict: Dict[str, Union[dict, float]], label_prefix = ''):
            for label, item in state_dict.items():
                if isinstance(item, dict):
                    _add_plot_entries(
                        state_dict[label],
                        label_prefix = label_prefix + label + '.')
                else:
                    full_label = label_prefix + label
                    if full_label not in self.available_plot_getters:
                        self.available_plot_getters[full_label] = _entry_data_getter(full_label)
                        self.available_plot_vars[full_label] = np.zeros(TRACE_LEN)

        self.available_plot_getters = {}
        self.available_plot_vars = {}
        _add_plot_entries(asdict(state))

        self.available_plot_labels = list(self.available_plot_getters.keys())
        self.selected_plot_vars = list(range(1, 1+NUMBER_OF_TRACES))

        self.available_abscissa_labels = list(self.available_plot_getters.keys())
        self.selected_abscissa_vars = [0] * NUMBER_OF_TRACES

    def update_camera_follow(self, state: BaseBodyState):
        ''' update camera follow for a state '''
        self.update_camera_pose(
            camera_follow_view_mat(
                self._camera_follow_modes[self._camera_follow_mode],
                state, self.dom)
        )

    def draw(self) -> bool:
        '''
        draw the window, both OpenGL and Imgui
        returns true while the window should stay open
        '''
        if not self.window:
            return True

        if self._recording_window:
            self._read_screen_capture()

        # gather events
        glfw.poll_events()
        if not self.window_open:
            # delay as if there were a refresh delay
            time.sleep(1/60)
            return self.should_close

        self.impl.process_inputs()

        # draw imgui
        if self.show_imgui:
            self._draw_imgui()
        else:
            # make an empty frame so callbacks work
            imgui.new_frame()
        self._process_mouse_drag()

        # draw opengl
        self._draw_opengl()

        # renger imgui on top of opengl
        imgui.render()
        if self.show_imgui:
            self.impl.render(imgui.get_draw_data())

        # push render to window
        glfw.swap_buffers(self.window)

        if self._recording_window:
            self._request_screen_capture()

        self._process_key_presses()

        return not self.should_close

    def add_object(self, name:str, obj: Renderable,
            show:bool = True,
            translucent: bool = False):
        ''' add an opengl style object to the window for drawing '''
        if translucent:
            assert name not in self.translucent_window_objects
            self.translucent_window_objects[name] = [show, obj]
        else:
            assert name not in self.window_objects
            self.window_objects[name] = [show, obj]

    def close(self):
        ''' close the window '''
        if not self.window:
            return
        self.window = None
        glfw.terminate()

    def start_recording(self, filename:Optional[str] = None):
        ''' start capturing window data as a video'''
        if not RECORDING_SUPPORTED:
            raise RuntimeError('opencv-python must be installed for window captures')

        project_folder = get_project_folder()
        rec_folder = os.path.join(project_folder, 'recordings')
        if not os.path.exists(rec_folder):
            os.mkdir(rec_folder)
        if filename is None:
            filename = 'window_recording.avi'
            k = 0
            while os.path.exists(os.path.join(rec_folder, filename)):
                k += 1
                filename = f'window_recording_{k}.avi'

        if not filename.endswith('.avi'):
            filename += '.avi'

        filename = os.path.join(rec_folder, filename)
        print(f'Saving Recording to {filename}')

        mode = glfw.get_video_mode(glfw.get_primary_monitor())

        self.create_recording_pbo()
        self._recording_window = True
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self._recording_writer = cv2.VideoWriter(filename, fourcc, mode.refresh_rate,
                                                 (self.window_width, self.window_height))

    def stop_recording(self):
        ''' finish capturing window data as a video '''
        assert self._recording_window
        self._recording_writer.release()
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        self._recording_window = False

    def create_recording_pbo(self):
        ''' register a pixel buffer object for screen captures '''
        if self._window_read_pbo is None:
            self._window_read_pbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._window_read_pbo)
        nbytes = self.window_height * self.window_width * 3
        gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, nbytes, ctypes.c_void_p(0), gl.GL_STREAM_READ)

    def _read_screen_capture(self):
        ''' starts a screen capture, does not return data '''
        ptr = gl.glMapBuffer(gl.GL_PIXEL_PACK_BUFFER, gl.GL_READ_ONLY)
        nbytes = self.window_height * self.window_width * 3
        if ptr:
            buf = (ctypes.c_int8 * nbytes).from_address(ptr)
            data: np.ndarray = np.frombuffer(buf, dtype = np.uint8).reshape(
                (self.window_height, self.window_width, 3)
            )
            data = np.flip(data, 0)

            self._recording_writer.write(data)
        gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)

    def _request_screen_capture(self):
        gl.glReadPixels(0,0,self.window_width, self.window_height,
                        gl.GL_BGR, gl.GL_UNSIGNED_BYTE, 0)

    def _create_imgui(self):
        imgui.create_context()

    def _create_window(self):
        if not glfw.init():
            self.window = None
            return

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        if platform.system() == 'Darwin':
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        glfw.window_hint(glfw.SAMPLES, 4)

        if self._fullscreen:
            video_mode = glfw.get_video_mode(glfw.get_primary_monitor())
            self.window_width = video_mode.size.width
            self.window_height = video_mode.size.height

        self.window = glfw.create_window(self.window_width,
                                         self.window_height,
                                         "Vehicle 3D",
                                         glfw.get_primary_monitor() if self._fullscreen else None,
                                         None)

        if not self.window:
            glfw.terminate()
            return

        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_window_size_callback(self.window, self._on_resize)

        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POLYGON_SMOOTH)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.window_width, self.window_height = glfw.get_window_size(self.window)

    def _on_resize(self, window, w, h):
        if self.window and window:
            if w == 0 or h == 0:
                # window has been minimized, but not closed
                self.window_open = False
                return
            self.window_open = True
            self.window_width = w
            self.window_height = h
            gl.glViewport(0,0,self.window_width,self.window_height)

            self.update_projection()

    def _create_imgui_renderer(self):
        if self.window:
            self.impl = GlfwRenderer(self.window)
            io = imgui.get_io()

            self.imgui_font = io.fonts.add_font_from_file_ttf(
                get_font_file('DejaVuSans.ttf'), 18)
            self.big_imgui_font = io.fonts.add_font_from_file_ttf(
                get_font_file('DejaVuSans.ttf'), 32)
            self.impl.refresh_font_texture()

    def _generate_default_objects(self, skybox = True, ground_plane = True):
        if self.dom is not None:
            surf = self.dom.triangulate(self.ubo)
            self.add_object('Surface', surf)

        if self.obstacles is not None:
            V, I = join_vis([
                obs.triangulate_obstacle() for obs in self.obstacles
            ])
            obstacles = VertexObject(self.ubo, V, I)
            self.add_object('Obstacles', obstacles)
            V, I = join_vis([
                obs.triangulate_obstacle(lines=True) for obs in self.obstacles
            ])
            obstacle_outlines = VertexObject(self.ubo, V, I, lines=True)
            self.add_object('Obstacle Outlines', obstacle_outlines)

        self._add_skybox(show = skybox)
        self._add_ground_plane(show=ground_plane)

    def _add_skybox(self, show = True):
        ''' add a skybox to the window '''
        skybox = Skybox()

        self.add_object('Skybox', skybox, show=show)
        self.skybox = skybox

    def _add_ground_plane(self, show=True):
        ''' add a ground plane to the window '''
        ground_plane = GroundPlane(self.ubo)
        self.add_object('Ground Plane', ground_plane, show)

    def update_projection(self):
        ''' update the camera projection for the window '''
        ps = glm.perspective(45.0, self.window_width / self.window_height, 0.1, 3000.0).T
        ps = ps.astype(np.float32)
        self._update_projection(ps)

    def _update_projection(self, ps: np.ndarray):
        self.ubo.update_projection(ps)
        self.skybox.update_projection(ps)

    def update_camera_pose(self, u_view=None):
        ''' update the camera pose transform for the window '''
        if u_view is None:
            u_view = np.eye(4)

        if self.camera_follow:
            u_camera = np.eye(4, dtype=np.float32)
            u_camera[:3,-1] = -self.camera_follow_x.to_vec()
            u_camera[:3,:3] = self.camera_follow_q.Rinv()

            u_view = u_camera @ u_view
        else:
            u_view[:3,-1] = -self.free_camera_q.Rinv() @ self.free_camera_x.to_vec()
            u_view[:3,:3] = self.free_camera_q.Rinv()

        u_view = u_view.astype(np.float32)
        self._update_view(u_view)

    def _update_view(self, u_view: np.ndarray):
        self.ubo.update_camera_pose(u_view)
        self.skybox.update_camera_pose(u_view)

    def _draw_opengl(self):
        # pylint: disable=unsupported-binary-operation
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_BLEND)
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

    def _draw_imgui(self):
        imgui.new_frame()
        imgui.push_font(self.imgui_font)

        imgui.set_next_window_position(self.window_width - self.imgui_width, 0)
        imgui.set_next_window_size(self.imgui_width, self.window_height)

        expanded = imgui.begin("Vehicle Info", closable = False,
            flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        if expanded:
            self.draw_vehicle_info()

        imgui.end()
        self._draw_camera_menu()
        self.draw_extras()

        imgui.pop_font()

    def draw_vehicle_info(self):
        ''' function to use or replace for drawing vehicle info '''
        # draw time series info on any provided states
        if self.available_plot_vars is None:
            return

        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()
        y0 = org.y + off.y
        plot_size = (0, int((self.window_height - y0)/NUMBER_OF_TRACES) - 52)

        generic_args = {
                'colors': [IMGUI_TEAL],
                'labels': ['Simulated Vehicle'],
                'size': plot_size,
                'show_title': False
            }

        for k in range(NUMBER_OF_TRACES):
            imgui.columns(2)

            _, self.selected_abscissa_vars[k] = \
                imgui.combo(f'##trace_x {k+1}',
                            self.selected_abscissa_vars[k],
                            self.available_abscissa_labels)
            imgui.next_column()
            _, self.selected_plot_vars[k] = \
                imgui.combo(f'##trace {k+1}',
                            self.selected_plot_vars[k],
                            self.available_plot_labels)
            imgui.columns(1)

            abs_label = self.available_abscissa_labels[self.selected_abscissa_vars[k]]
            var_label = self.available_plot_labels[self.selected_plot_vars[k]]
            show_tooltip = abs_label in ['t', 'p.s']
            tooltip_label = 's' if abs_label == 'p.s' else 't'

            x = [self.available_plot_vars[abs_label]]
            y = [self.available_plot_vars[var_label]]

            plot_multiline(
                time = x,
                data = y,
                current_time = [0],
                current_data = [0],
                title = f'##trace plot {k}',
                show_tooltip = show_tooltip,
                tooltip_label = tooltip_label,
                **generic_args)

    def _draw_camera_menu(self):
        imgui.set_next_window_position(self.window_width - self.imgui_width - 200, 0)
        imgui.set_next_window_size(200, 0) # autosize y
        expanded, _ = imgui.begin("Camera Settings",
            closable = False,
            flags = imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_SCROLLBAR)
        if expanded:
            if imgui.button("Reset Camera"):
                self._reset_camera()
            if imgui.radio_button('Camera Follow', self.camera_follow):
                self.camera_follow = not self.camera_follow
            if self.camera_follow:
                imgui.push_item_width(-1)
                _, self._camera_follow_mode = imgui.combo(
                    "##Selected Follow Mode",
                    self._camera_follow_mode,
                    self._camera_follow_mode_labels)
                imgui.pop_item_width()

            imgui.separator()
            imgui.text('Hide / Show:')
            # buttons to hide/show rendered objects
            for obj_name, (draw_obj, _) in self.window_objects.items():
                if imgui.radio_button(obj_name, draw_obj):
                    self.window_objects[obj_name][0] = not draw_obj
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

    def draw_extras(self):
        ''' function to replace for drawing extra items '''

    def _reset_camera(self):
        self.free_camera_x = Position()
        self.free_camera_q = OrientationQuaternion()
        if self.dom is not None:
            self.free_camera_q.from_vec([0.5,0.3,0.3,0.7])
            self.free_camera_q.normalize()
            self.free_camera_x.from_vec(
                (self.dom.view_center + self.dom.view_scale * self.free_camera_q.e3())
            )
        self.camera_follow_x = Position(xk = 12)
        if self.dom is not None:
            self.camera_follow_x.xk = self.dom.view_scale / 10
        self.camera_follow_q = OrientationQuaternion()
        self.camera_follow_q.from_vec([-0.3,0.3,0.6,-0.6])
        self.camera_follow_q.normalize()

    def _process_mouse_drag(self):
        active_drag = self._drag_mouse >= 0
        io = imgui.get_io()
        if not io.want_capture_mouse:
            self._update_scroll_mouse_drag(0., -10*io.mouse_wheel)

        if not active_drag:
            for mouse in self._drag_mice:
                if imgui.core.is_mouse_clicked(mouse):
                    if not io.want_capture_mouse:
                        self._drag_mouse = mouse
                        self._mouse_drag_prev_delta = (0,0)
        else:
            if imgui.core.is_mouse_released(self._drag_mouse):
                self._drag_mouse = -1
                return

            drag = imgui.get_mouse_drag_delta(self._drag_mouse)
            dx = drag[0] - self._mouse_drag_prev_delta[0]
            dy = drag[1] - self._mouse_drag_prev_delta[1]
            self._mouse_drag_prev_delta = drag

            self._drag_mouse_callbacks[self._drag_mouse](dx, dy)

        if not self.camera_follow:
            self.update_camera_pose(None)

    def _update_right_mouse_drag(self, dx, dy):
        w = BodyAngularVelocity(w2 = -dx, w1 = -dy, w3 = 0)
        if self.camera_follow:
            qdot = self.camera_follow_q.qdot(w)
            self.camera_follow_q.qr += qdot.qr * 0.002
            self.camera_follow_q.qi += qdot.qi * 0.002
            self.camera_follow_q.qj += qdot.qj * 0.002
            self.camera_follow_q.qk += qdot.qk * 0.002
            self.camera_follow_q.normalize()
        else:
            qdot = self.free_camera_q.qdot(w)
            self.free_camera_q.qr += qdot.qr * 0.002
            self.free_camera_q.qi += qdot.qi * 0.002
            self.free_camera_q.qj += qdot.qj * 0.002
            self.free_camera_q.qk += qdot.qk * 0.002
            self.free_camera_q.normalize()

    def _update_left_mouse_drag(self, dx, dy):
        if self.camera_follow:
            self.camera_follow_x.xi -= dx / 150 * max(1, abs(self.camera_follow_x.xk / 20))
            self.camera_follow_x.xj += dy / 150 * max(1, abs(self.camera_follow_x.xk / 20))
        else:
            scale = self.dom.view_scale/1000 if self.dom is not None else 5
            self.free_camera_x.from_vec(
                self.free_camera_x.to_vec() +
                (-dx*self.free_camera_q.e1() + dy*self.free_camera_q.e2()) * scale
            )

    def _update_scroll_mouse_drag(self, _, dy):
        if self.camera_follow:
            if abs(self.camera_follow_x.xk) > 3:
                self.camera_follow_x.xk += dy * self.camera_follow_x.xk / 100
            else:
                self.camera_follow_x.xk += dy / 30
            self.camera_follow_x.xk = max(0, self.camera_follow_x.xk)
        else:
            scale = self.dom.view_scale/200 if self.dom is not None else 25
            self.free_camera_x.from_vec(
                self.free_camera_x.to_vec() +
                dy*self.free_camera_q.e3() * scale
            )

    def _process_key_presses(self):
        if glfw.window_should_close(self.window):
            self.should_close = True
        elif self.impl.io.keys_down[glfw.KEY_ESCAPE]:
            self.should_close = True

        io = imgui.get_io()
        if not io.want_capture_keyboard:
            if self.impl.io.keys_down[glfw.KEY_SPACE]:
                if self._last_show_imgui_change is None or \
                        time.time() - self._last_show_imgui_change > 0.3:
                    self.show_imgui = not self.show_imgui
                    self._last_show_imgui_change = time.time()

            if self.impl.io.keys_down[glfw.KEY_R]:
                if self._last_record_window_change is None or \
                        time.time() - self._last_record_window_change > 1.0:
                    if not self._recording_window:
                        self.start_recording()
                    else:
                        self.stop_recording()
                    self._last_record_window_change = time.time()
