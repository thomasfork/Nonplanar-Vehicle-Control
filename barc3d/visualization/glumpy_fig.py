import numpy as np
from glumpy import app, gl, glm, gloo, transforms
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
#from glumpy.transforms import Position, OrthographicProjection, Viewport

import glumpy
import scipy.linalg
import os


from multiprocessing import Process, Queue, Value

from barc3d.visualization.base_fig import BaseFigure

from barc3d.pytypes import VehicleState, VehicleConfig, Position, OrientationQuaternion, BodyAngularVelocity
from barc3d.surfaces.base_surface import BaseSurface
from barc3d.visualization.utils import load_ford_mustang
from barc3d.visualization import shaders

class GlumpyFig(BaseFigure):
    def __init__(self, surf: BaseSurface, config: VehicleConfig, state:VehicleState = VehicleState()):
        
        self.f_queue = Queue()
        self.fig_ready = Value('i',0)
        self.manager = DriveWindowManager(self.f_queue, self.fig_ready)
        
        
        self.subp = Process(target = self.manager.run, args = (surf, config, state))
        self.subp.start()
        return
    
    def available(self = None):
        for backend in app.window.backends.__backends__:
            if app.use(backend):
                return True
        return False
        
    def ready(self):
        return self.fig_ready.value
    
    def draw(self, state: VehicleState):
        self.f_queue.put(state.p)
        return self.subp.is_alive()
    
    def close(self):
        self.fig_ready.value = 0
        return
    

class DriveWindowManager():
    '''
    class meant for running the window using a subprocess
    '''
    
    def __init__(self,f_queue:Queue, fig_ready:Value):
        self.f_queue = f_queue
        self.fig_ready = fig_ready
        self.fig_ready.value = 0
        return
    
    
    def run(self, surf: BaseSurface, config: VehicleConfig, state:VehicleState = VehicleState()):
        self.startup(surf, config, state)
        self.loop()
        return
    
    def startup(self, surf: BaseSurface, config: VehicleConfig, state:VehicleState = VehicleState()):
    
        self.state = state
        self.drive_window = DriveWindow()
        self.surf = surf
        
        V,I = surf.generate_texture(road_offset = 0 if config.road_surface else -config.h, as_opengl = True)
        self.road = DriveWindowTexture(V,I)
        
        V,I = load_ford_mustang(config)
        self.car = DriveWindowTexture(V,I)
    
        self.drive_window.add_texture('road',self.road)
        self.drive_window.add_texture('car', self.car)
        
        self.backend = app.__backend__
        self.clock = app.__init__(backend = self.backend)
        self.count = len(self.backend.windows())
        
        self.fig_ready.value = 1
        return
    
        
    def loop(self):
        while self.count and self.fig_ready.value: 
            msg = ''
            while not self.f_queue.empty():
                self.state.p = self.f_queue.get()
            
            self.surf.local_to_global(self.state)
                
            self.count = self.backend.process(self.clock.tick())
            self.car.update_pose(self.state.x, self.state.q)
            
            self.drive_window.update_camera_pose(self.state.x, self.state.q)
            
        self.drive_window.close()
        app.quit()
        return
         
        
    

       
class DriveWindow():
    def __init__(self):
        config = app.configuration.Configuration()
        config.samples = 8
        window = app.Window(width=1024, height=1024,
                        color=(1, 1, 1, 1.00),
                        config = config)
                         

        self.window = window       
        self.textures = dict()
        
        self.vehicle_x = Position()
        self.vehicle_q = OrientationQuaternion()
        
        self.q = OrientationQuaternion()
        self.zoom = 80
        self.xi_offset = 0
        self.xj_offset = 0
        
        self.camera_mode = 0
        
        self.view_1_x = Position(xi=40, xj=19, xk=0)
        self.view_1_q = OrientationQuaternion(qr=0.8803637210121743, qi=-0.15564045298582307, qj=0.43832921224705945, qk=0.0927538129180608)
        self.view_1_zoom = 80
        
        self.view_2_x = Position(xi=42, xj=115, xk=0)
        self.view_2_q = OrientationQuaternion(qr=-0.5332696885613157, qi=0.17594304778502606, qj=0.7868527694614451, qk=0.255988676289962)
        self.view_2_zoom = 80
        
        @window.event
        def on_draw(dt):
            window.clear()
            for texture_name in self.textures:
                self.textures[texture_name].draw()
                
                
            return

        @window.event
        def on_resize(width, height):
            for texture_name in self.textures:
                self.textures[texture_name].texture['u_projection'] = glm.perspective(45.0, width / float(height), 10.0, 1000.0)

        @window.event
        def on_init():
            gl.glEnable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glEnable(gl.GL_POLYGON_SMOOTH)
            gl.glEnable(gl.GL_MULTISAMPLE)
        
        @window.event
        def on_mouse_drag(x, y, dx, dy, button):
            w = BodyAngularVelocity(w2 = -dx, w1 = -dy, w3 = 0)
            qdot = self.q.qdot(w)
            self.q.qr += qdot.qr * 0.001
            self.q.qi += qdot.qi * 0.001
            self.q.qj += qdot.qj * 0.001
            self.q.qk += qdot.qk * 0.001
            self.q.normalize()
            
            
        @window.event
        def on_mouse_scroll(x, y, dx, dy):
            self.zoom -= dy / 5
            
            w = BodyAngularVelocity(w2 = 0, w1 = 0, w3 = dx)
            qdot = self.q.qdot(w)
            self.q.qr += qdot.qr * 0.05
            self.q.qi += qdot.qi * 0.05
            self.q.qj += qdot.qj * 0.05
            self.q.qk += qdot.qk * 0.05
            self.q.normalize()
            
        @window.event
        def on_key_press(symbol, modifiers):
            inc = 1
            if   symbol == 65362:    # up arrrow key
                self.xi_offset += inc
            elif symbol == 65364:    # down arrow key
                self.xi_offset -= inc
            elif symbol == 65363:    # right arrow key
                self.xj_offset += inc
            elif symbol == 65361:    # left arrow key
                self.xj_offset -= inc
            elif symbol == 65293:    # enter key
                print('Current Camera State:')
                x = Position() #x.copy()
                x.xi = self.xi_offset
                x.xj = self.xj_offset
            elif symbol == 32:       # spacebar
                self.xi_offset = 0
                self.xj_offset = 0
                self.q = OrientationQuaternion()
                self.zoom = 20
            elif symbol == 65289:    # tab
                self.camera_mode += 1
                self.camera_mode %= 4
                return
            return
    
    
    def close(self):
        try:
            self.window.close()
        except ValueError: #may have already been closed
            return
            
    def add_texture(self, name:str, texture: 'DriveWindowTexture'):
        self.textures[name] = texture
    
        
    def update_camera_pose(self, x, q):
        self.vehicle_x = x
        self.vehicle_q = q
        if self.camera_mode == 0:
            for texture_name in self.textures:
                
                self.textures[texture_name].update_camera_follow_pose(x, q)
        elif self.camera_mode == 1:
            x = Position() #x.copy()
            x.xi = self.xi_offset
            x.xj = self.xj_offset
            for texture_name in self.textures:
                self.textures[texture_name].update_camera_pose(x, self.q, self.zoom)
        elif self.camera_mode == 2:
            for texture_name in self.textures:
                self.textures[texture_name].update_camera_pose(self.view_1_x, self.view_1_q, self.view_1_zoom)
        elif self.camera_mode == 3:
            for texture_name in self.textures:
                self.textures[texture_name].update_camera_pose(self.view_2_x, self.view_2_q, self.view_2_zoom)
        return
      
    
class DriveWindowTexture():
    def __init__(self, V, I, O = None):
        vertex = shaders.vertex_default
        fragment = shaders.fragment_default
            
        
        V = V.view(gloo.VertexBuffer)
        I = I.view(gloo.IndexBuffer)
        if O is not None:
            O = O.view(gloo.IndexBuffer)
        
        texture = gloo.Program(vertex, fragment)
        texture['a_position'] = V['a_position']
        texture['a_normal'] = V['a_normal']
        texture['a_color'] = V['a_color']
        #texture.bind(V)
        texture['u_model'] = np.eye(4, dtype=np.float32)
        u_view = np.eye(4, dtype=np.float32)
        u_view = glm.rotate(u_view, 90, 0, 1, 0)
        u_view = glm.rotate(u_view, 90, 0, 0, 1)
        u_view = glm.rotate(u_view, 20, 1, 0, 0)
        u_view = glm.translate(u_view, 0,0,-20)
        texture['u_view'] = u_view
        
        self.texture = texture
        self.V = V
        self.I = I
        self.O = O
        return
    
    def update_pose(self, x, q): 
        ''' meant for moving the car around, not intended for moving a surface'''
        u_model = np.eye(4, dtype=np.float32)
        u_model = u_model @ scipy.linalg.block_diag(q.Rinv(), 1)
        u_model = glm.translate(u_model, x.xi, x.xj, x.xk)
        self.texture['u_model'] = u_model.astype(np.float32)
        return
        
    def update_camera_follow_pose(self, x, q, zoom = 20):
        u_view = np.eye(4, dtype=np.float32)
        u_view = glm.translate(u_view, -x.xi, -x.xj, -x.xk)
        u_view =  u_view @ scipy.linalg.block_diag(q.R(),1)
        
        
        u_view = glm.rotate(u_view, 90, 0, 1, 0)
        u_view = glm.rotate(u_view, 90, 0, 0, 1)
        u_view = glm.rotate(u_view, 35, 1,0,0)
        u_view = glm.translate(u_view, 0,0,-zoom)
        self.texture['u_view'] = u_view
        
        return
    
    def update_camera_pose(self, x, q, zoom = 20):
        u_view = np.eye(4, dtype=np.float32)
        u_view = glm.translate(u_view, -x.xi, -x.xj, -x.xk)
        u_view = glm.rotate(u_view, 90, 0, 1, 0)
        u_view = glm.rotate(u_view, 90, 0, 0, 1)
        
        u_view = u_view @ scipy.linalg.block_diag(q.R(),1)
        u_view = glm.translate(u_view, 0,0,-zoom)
        self.texture['u_view'] = u_view

    def draw(self):
        if self.I is not None:
            self.texture.draw(gl.GL_TRIANGLES, self.I)
        elif self.O is not None:
            self.texture.draw(gl.GL_LINES, self.O)
        
        return

