import numpy as np
import scipy.linalg
import ctypes
from collections import deque
import os
import time
from PIL import Image

import glfw
import OpenGL.GL as gl
import OpenGL.GL.shaders

from barc3d.visualization import glm

import imgui
from imgui.integrations.glfw import GlfwRenderer

from dataclasses import dataclass, field


import barc3d.visualization.shaders  as shaders
from multiprocessing import Process, Queue, Value

from barc3d.visualization.base_fig import BaseFigure

from barc3d.pytypes import VehicleState, VehicleConfig, Position, OrientationQuaternion, BodyAngularVelocity
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.visualization.utils import load_ford_mustang

BUF_LENGTH = 500

class OpenGLFig(BaseFigure):
    def __init__(self, surf: BaseSurface, config: VehicleConfig, state:VehicleState = VehicleState()):
        
        self.manager = WindowManager()
        
        self.subp = Process(target = self.manager.run, args = (surf, config, state))
        self.subp.start()
        return
        
    def ready(self):
        return self.manager.fig_ready.value
    
    def draw(self, state: VehicleState):
        self.manager.state_queue.put(state)
        return self.subp.is_alive()
    
    def close(self):
        self.manager.fig_ready.value = 0
        return
    
    def set_camera_follow(self, val = False):
        self._safe_queue_put('camera_follow_%d'%(1 if val else 0)) 
       
    def _safe_queue_put(self, msg):
        try:
            self.manager.input_command_queue.put(msg)
        except BrokenPipeError:
            self.manager.fig_ready.value = -1
        return  
        
         
class WindowManager():
    '''
    class meant for running the window using a subprocess
    '''
    
    def __init__(self):
        self.state_queue = Queue()
        self.input_command_queue = Queue()
        self.fig_ready = Value('i',0)
        self.fig_ready.value = 0
        return
    
    
    def run(self, surf: BaseSurface, config: VehicleConfig, state:VehicleState = VehicleState()):
        self.startup(surf, config, state)
        self.loop()
        return
    
    def startup(self, surf: BaseSurface, config: VehicleConfig, state:VehicleState = VehicleState()):
    
        self.state = state
        self.window = Window()  
        self.surf = surf
        
        V,I = surf.generate_texture(road_offset = 0 if config.road_surface else -config.h, as_opengl = True)
        self.road = OpenGLObject(V,I)
        
        V,I = load_ford_mustang(config)
        self.car = OpenGLObject(V,I)
    
        self.window.add_object('road',self.road)
        self.window.add_object('car', self.car)
        self.window.update_projection()
        
        
        self.fig_ready.value = 1
        return
    
    def loop(self):
        while self.fig_ready.value:
            new_state = False
            while not self.state_queue.empty():
                self.state = self.state_queue.get()
                new_state = True
            while not self.input_command_queue.empty():
                msg = self.input_command_queue.get()
                if isinstance(msg, str):
                    self._process_string_command(msg)   
            self.car.update_pose(self.state.x, self.state.q)
            
            self.window.update_camera_pose(self.state.x, self.state.q)
            
            should_close = self.window.draw(self.state, new_state)
            if should_close:
                self.fig_ready.value = 0
        self.window.close()
        return
        
    def process_string_command(self, msg):
        if msg == 'camera_follow_0':
            self.window.camera_follow = False
        elif msg == 'camera_follow_1':
            self.window.camera_follow = True

class Window():
    def __init__(self):
        self.window = None
        self.window_width = 1920
        self.window_height = 1080
        self.window_objects = dict()
        self.window_resized = False
        
        self.impl = None
        
        self.state_buf = deque([VehicleState()]*BUF_LENGTH)
        
        self.should_close = False
        self.camera_follow = True
        self.camera_follow_x = Position()
        self.camera_follow_q = OrientationQuaternion()
        self.camera_follow_z = 20
        self.mouse_drag_prev_delta = (0,0)
        self.drag_mice = [0,1,2] # left, right, scroll wheel
        self.drag_mouse = -1 # -1: no mouse drag currently happening
        self.drag_mouse_callbacks = {0:self._update_left_mouse_drag, 
                                     1:self._update_right_mouse_drag,
                                     2:self._update_scroll_mouse_drag}
        
        self._create_imgui()
        self._create_window()
        self._create_imgui_renderer()
        return
           
    def add_object(self, name, obj):
        assert isinstance(obj, OpenGLObject)
        assert name not in self.window_objects
        self.window_objects[name] = obj
        return
    
    def draw(self, state = None, new_state = False):
        if not self.window:
            return
       
        if state is not None and new_state:
            self.state_buf.append(state)
            self.state_buf.popleft()
        
        # gather events    
        glfw.poll_events()
        self.impl.process_inputs()
        
        # draw imgui   
        self._draw_imgui()
            
        # draw opengl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        for name in self.window_objects:
            self.window_objects[name].draw()
        
        # renger imgui on top of opengl    
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        
        # push render to window
        glfw.swap_buffers(self.window)
        
        if glfw.window_should_close(self.window):
            self.should_close = True
            
        return self.should_close
    
    def close(self):
        if not self.window:
            return
        self.window = None
        glfw.terminate()
        return
    
    def update_projection(self):
        ps = glm.perspective(45.0, self.window_width / self.window_height, 1.0, 300.0)
        for name in self.window_objects:
            self.window_objects[name]._update_u_projection(ps)
        return   
    
    def update_camera_pose(self, x, q, zoom = 30):
        if self.camera_follow:
            u_view = np.eye(4, dtype=np.float32)
            
            u_view = glm.translate(u_view, -x.xi, -x.xj, -x.xk)
            u_view =  u_view @ scipy.linalg.block_diag(q.R(),1)
            u_view = glm.rotate(u_view, 90, 0, 1, 0)
            u_view = glm.rotate(u_view, 90, 0, 0, 1)
            u_view = glm.rotate(u_view, 35, 1,0,0) 
            u_view = glm.translate(u_view, 0,0,-zoom)
            
            for name in self.window_objects:
                self.window_objects[name].update_camera_pose(u_view.astype(np.float32))
        else:
            u_view = np.eye(4, dtype=np.float32)
            u_view = glm.translate(u_view, -self.camera_follow_x.xi, -self.camera_follow_x.xj, -self.camera_follow_x.xk)
            
            u_view =  u_view @ scipy.linalg.block_diag(self.camera_follow_q.R(),1)
            u_view = glm.translate(u_view, 0,0,-self.camera_follow_z)
            
            for name in self.window_objects:
                self.window_objects[name].update_camera_pose(u_view.astype(np.float32))    
        return
    
    def _create_imgui(self):
        imgui.create_context()
        
    def _create_window(self):
        if not glfw.init():
            self.window = None
            return
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            
        glfw.window_hint(glfw.SAMPLES, 8)
        fullscreen = False
        self.window = glfw.create_window(self.window_width, self.window_height, "Barc3D", glfw.get_primary_monitor() if fullscreen else None, None)
        
        if not self.window:
            glfw.terminate()
            return
        
        glfw.make_context_current(self.window)
        glfw.set_window_size_callback(self.window, self._on_resize)       # this captures programmatically resizing the window
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)  # this captures manually resizing the window
        
        
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glEnable(gl.GL_POLYGON_SMOOTH)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glLineWidth(3)
        
        return
    
    def _get_font_file(self, name = 'DejaVuSans.ttf'):
        folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts')
        
        filename = os.path.join(folder, name)
        return filename
        
    
    def _create_imgui_renderer(self):
        if self.window:
            self.impl = GlfwRenderer(self.window)
            io = imgui.get_io()
            
            self.imgui_font = io.fonts.add_font_from_file_ttf(
                self._get_font_file('DejaVuSans.ttf'), 18)
            self.big_imgui_font = io.fonts.add_font_from_file_ttf(
                self._get_font_file('DejaVuSans.ttf'), 32)
            self.impl.refresh_font_texture()
            
    def _on_resize(self, window, w, h):
        if self.window:
            self.window_width = w
            self.window_height = h  
            gl.glViewport(0,0,self.window_width,self.window_height)
            
            self.update_projection()
        return
        
    def _draw_imgui(self):
        imgui.new_frame()
        imgui.push_font(self.imgui_font)
        
        self._process_mouse_drag()
        
        
        imgui.set_next_window_position(self.window_width - 300, 0)
        imgui.set_next_window_size(300,900)
             
        imgui.begin("Vehicle History", closable = False)
            
        v = np.asanyarray([s.v.signed_mag() for s in self.state_buf]).astype(np.float32)
        y = np.asanyarray([s.p.y for s in self.state_buf]).astype(np.float32)
        ths = np.asanyarray([s.p.ths for s in self.state_buf]).astype(np.float32)
        N = np.asanyarray([s.fb.f3 for s in self.state_buf]).astype(np.float32)
        ua = np.asanyarray([s.u.a for s in self.state_buf]).astype(np.float32)
        uy = np.asanyarray([s.u.y for s in self.state_buf]).astype(np.float32)
            
        if imgui.radio_button('Camera Follow', self.camera_follow):
            self.camera_follow = not self.camera_follow
        imgui.text('Vehicle Speed: %0.2f m/s'%self.state_buf[-1].v.signed_mag())
             
        imgui.plot_lines('',
                         v,
                         overlay_text = 'Speed',
                         graph_size = (280,100))
        imgui.plot_lines('',
                         y,
                         overlay_text = 'Lane Offset',
                         graph_size = (280,100))
        imgui.plot_lines('',
                         ths,
                         overlay_text = 'Heading',
                         graph_size = (280,100))
        imgui.plot_lines('',
                         N,
                         overlay_text = 'Normal Force',
                         graph_size = (280,100))
        imgui.plot_lines('',
                         ua,
                         overlay_text = 'Throttle',
                         graph_size = (280,100))
        imgui.plot_lines('',
                         uy,
                         overlay_text = 'Steering',
                         graph_size = (280,100))
        imgui.end()
            
        imgui.pop_font()
        return
    
    def _process_mouse_drag(self):
        if self.camera_follow:
            self.drag_mouse = -1
            return
            
        active_drag = self.drag_mouse >= 0
        
        if not active_drag:
            for mouse in self.drag_mice:
                if imgui.core.is_mouse_clicked(mouse):
                    self.drag_mouse = mouse
                    self.mouse_drag_prev_delta = (0,0)
        else:
            if imgui.core.is_mouse_released(self.drag_mouse):
                self.drag_mouse = -1 
                return
                
            drag = imgui.get_mouse_drag_delta(self.drag_mouse)
            dx = drag[0] - self.mouse_drag_prev_delta[0]
            dy = drag[1] - self.mouse_drag_prev_delta[1]
            self.mouse_drag_prev_delta = drag
            
            self.drag_mouse_callbacks[self.drag_mouse](dx, dy)
            self.update_camera_pose(None, None)
        return

    def _update_right_mouse_drag(self, dx, dy):
        w = BodyAngularVelocity(w2 = -dx, w1 = -dy, w3 = 0)
        qdot = self.camera_follow_q.qdot(w)
        self.camera_follow_q.qr += qdot.qr * 0.002
        self.camera_follow_q.qi += qdot.qi * 0.002
        self.camera_follow_q.qj += qdot.qj * 0.002
        self.camera_follow_q.qk += qdot.qk * 0.002
        self.camera_follow_q.normalize()
        return
        
    def _update_left_mouse_drag(self, dx, dy):
        d1 =-self.camera_follow_q.e1() * dx * self.camera_follow_z / 1500
        d2 = self.camera_follow_q.e2() * dy * self.camera_follow_z / 1500
        
        self.camera_follow_x.xi += d1[0] + d2[0]
        self.camera_follow_x.xj += d1[1] + d2[1]
        self.camera_follow_x.xk += d1[2] + d2[2]
        return
        
    def _update_scroll_mouse_drag(self, dx, dy): 
        d1 = self.camera_follow_q.e3() * dy * self.camera_follow_z / 300
        
        self.camera_follow_x.xi += d1[0] 
        self.camera_follow_x.xj += d1[1] 
        self.camera_follow_x.xk += d1[2] 
        return
    
    
class OpenGLObject():
    def __init__(self, V, I = None, simple = False, static_draw = True, lines = False):
        self.vertex = shaders.vertex_default
        self.fragment = shaders.fragment_default if not simple else shaders.fragment_simple
        self.program = None
        self.static_draw = static_draw
        self.lines = lines
        
        self.get_program()
        
        self.setup(V,I)
        
        self._update_u_model(np.eye(4).astype(np.float32))
        self._update_u_view(np.eye(4).astype(np.float32))
        self._update_u_projection(np.eye(4).astype(np.float32))
        return
   
    def get_program(self):
        self.program = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(self.vertex,   gl.GL_VERTEX_SHADER),
                                                       OpenGL.GL.shaders.compileShader(self.fragment, gl.GL_FRAGMENT_SHADER))
        gl.glUseProgram(self.program)
        
        VAO = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(VAO)
        VBO = gl.glGenBuffers(1)
        
        self.VAO = VAO
        self.VBO = VBO
        return
    
    def setup(self, vertices, indices):
        if not self.program:
            return
        
        if indices is not None:
            vertices = vertices[indices]
        
        self.no_indices = vertices.shape[0]
        
        
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.VAO)
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.VBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW if self.static_draw else gl.GL_DYNAMIC_DRAW)
        
        position = gl.glGetAttribLocation(self.program, 'a_position')
        gl.glVertexAttribPointer(position, 3, gl.GL_FLOAT, gl.GL_FALSE, 40, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(position)
        
        normal = gl.glGetAttribLocation(self.program, 'a_normal')
        if normal > 0: # this attribute is not active for the simple fragment
            gl.glVertexAttribPointer(normal, 3, gl.GL_FLOAT, gl.GL_FALSE, 40, ctypes.c_void_p(12))
            gl.glEnableVertexAttribArray(normal)
            
        color = gl.glGetAttribLocation(self.program, 'a_color')
        gl.glVertexAttribPointer(color, 4, gl.GL_FLOAT, gl.GL_FALSE, 40, ctypes.c_void_p(24))
        gl.glEnableVertexAttribArray(color)
        
        gl.glUseProgram(0)
        return
   
    def draw(self):
        if not self.program:
            return
            
        gl.glUseProgram(self.program)
        gl.glBindVertexArray(self.VAO)
        if self.lines:
            gl.glDrawArrays(gl.GL_LINES, 0, self.no_indices)
        else:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.no_indices)
        gl.glUseProgram(0)
        return
    
    def update_pose(self, x, q): 
        ''' meant for moving the car around, not intended for moving a surface'''
        u_model = np.eye(4, dtype=np.float32)
        u_model = u_model @ scipy.linalg.block_diag(q.Rinv(), 1)
        u_model = glm.translate(u_model, x.xi, x.xj, x.xk)
        self._update_u_model(u_model.astype(np.float32))
        
        return
    
    def update_camera_pose(self, mat):
        self._update_u_view(mat)
        return
    
    def update_projection(self, mat):
        self.u_projection = mat.copy()
        return
    
    def _update_u_model(self, u_model):
        if not self.program:
            return
        gl.glUseProgram(self.program)
        transformLoc = gl.glGetUniformLocation(self.program, "u_model")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, u_model)
        gl.glUseProgram(0)
        return
        
    def _update_u_view(self, u_view):
        if not self.program:
            return
        gl.glUseProgram(self.program)
        transformLoc = gl.glGetUniformLocation(self.program, "u_view")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, u_view)
        gl.glUseProgram(0)
        return
    
    def _update_u_projection(self, u_projection):
        if not self.program:
            return
        gl.glUseProgram(self.program)
        transformLoc = gl.glGetUniformLocation(self.program, "u_projection")
        gl.glUniformMatrix4fv(transformLoc, 1, gl.GL_FALSE, u_projection)
        gl.glUseProgram(0)
        return

