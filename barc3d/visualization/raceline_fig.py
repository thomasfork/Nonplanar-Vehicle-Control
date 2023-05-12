import numpy as np
import imgui
import glfw
import time

from barc3d.pytypes import VehicleState

from barc3d.visualization.utils import load_trajectory, load_ford_mustang, load_circle, get_cmap_rgba, plot_multiline
from barc3d.visualization.opengl_fig import Window, OpenGLObject 

import time

class RacelineWindow(Window):
    results = None
    
    def __init__(self, surf, model, results, obstacles = None, circular_buf = None, run = True, closed = None):
        if not isinstance(results, list):
            results = [results]
        
        super().__init__()
        
        self.surf = surf
        self.model = model
        self.results = results
        self.obstacles = obstacles
        self.circular_buf = circular_buf
        self.closed = closed if closed is not None else self.surf.config.closed
        self.cars = []
        self.animation_t = 0
        self.animation_t0 = None
        self.animation_dt = 1
        
        self.running = True
        self.reverse = False
        
        self.camera_follow_z = self.model.L*7
        
        self._precompute_parameters()
        self._generate_objects()
        if run:
            self.run()
        return
    
    
    def run(self):
        while not self.should_close:
            self.draw()
        self.close()
        return
        
    def _precompute_parameters(self):
        self.t = np.array([[state.t for state in line.raceline] for line in self.results])
        self.s = np.array([[state.p.s for state in line.raceline] for line in self.results])
        
        self.y = np.array([[state.p.y for state in line.raceline] for line in self.results])
        self.ths = np.array([[state.p.ths for state in line.raceline] for line in self.results])
        
        self.v1 = np.array([[state.v.v1 for state in line.raceline] for line in self.results])
        self.v2 = np.array([[state.v.v2 for state in line.raceline] for line in self.results])
        self.v  = np.array([[state.v.signed_mag() for state in line.raceline] for line in self.results])
        
        self.w1 = np.array([[state.w.w1 for state in line.raceline] for line in self.results])
        self.w2 = np.array([[state.w.w2 for state in line.raceline] for line in self.results])
        self.w3 = np.array([[state.w.w3 for state in line.raceline] for line in self.results])
        
        self.ua = np.array([[state.u.a for state in line.raceline] for line in self.results])
        self.uy = np.array([[state.u.y for state in line.raceline] for line in self.results])
        
        self.colors = [imgui.get_color_u32_rgba(*line.color) for line in self.results]
        self.labels = [line.label for line in self.results]
        self.times  = [line.time for line in self.results]
        
        self.t_min = self.t.min()
        self.t_max = self.t.max()
        self.v_min = self.v.min()
        self.v_max = self.v.max()
        
        
    def _generate_objects(self):
        s_ext = 0 if self.surf.config.closed else self.model.L
        line_w = self.model.tf/3
        
        V,I = self.surf.generate_texture(y_ext = self.model.lf, s_ext = s_ext)  
        road = OpenGLObject(V,I)    
        self.add_object('road',road)
        
        for k,line in enumerate(self.results):
            V, I = load_trajectory(self.surf, line.raceline, road_offset = self.model.h/60*(k+3), h = self.model.h / 5, w = line_w, 
                                   v_min = self.v_min, v_max = self.v_max, closed = self.closed)
            raceline_texture  = OpenGLObject(V,I, simple = True)
            self.add_object('raceline_%d'%k, raceline_texture)
            
            V,I = load_ford_mustang(self.model.vehicle_config, line.color)
            car  = OpenGLObject(V,I)
            car.update_pose(line.raceline[0].x, line.raceline[0].q)
            
            self.add_object('car_%d'%k, car)
            
            if self.circular_buf is not None:
                V,I = load_circle(h = self.model.h, t = self.model.h/2, ro = self.circular_buf, C = line.color)
                circle = OpenGLObject(V,I, simple = True)
                self.add_object('circle_%d'%k, circle)
                circle.update_pose(line.raceline[0].x, line.raceline[0].q)  
                car.circular_buffer_object = circle
                
            self.cars.append(car)
        
        
        if self.obstacles:
            V_obs, I_obs = None, None
            V_obs_outline, I_obs_outline = None, None
            for obs in self.obstacles:
                V, I = obs.generate_texture(lines = False)
                
                if V_obs is None:
                    V_obs = V
                    I_obs = I
                else:
                    I_obs = np.concatenate([I_obs, I + V_obs.shape[0]])
                    V_obs = np.concatenate([V_obs, V])
                    
                V, I = obs.generate_texture(lines = True)
                if V_obs_outline is None:
                    V_obs_outline = V
                    I_obs_outline = I
                else:
                    I_obs_outline = np.concatenate([I_obs_outline, I + V_obs_outline.shape[0]])
                    V_obs_outline = np.concatenate([V_obs_outline, V])
                
                
            I_obs = I_obs.astype(np.uint32)
            I_obs_outline = I_obs_outline.astype(np.uint32)
            
            obstacles = OpenGLObject(V_obs, I_obs)
            self.add_object('obstacles', obstacles)
            
            obstacle_outlines = OpenGLObject(V_obs_outline, I_obs_outline, lines = True)
            self.add_object('obstacle_outlines', obstacle_outlines)
        
        self.update_projection()
        self.camera_follow = True
        self.update_camera_pose(self.results[-1].raceline[0].x, self.results[-1].raceline[0].q, zoom = self.model.L*13)
        return

    def _draw_imgui(self):
        imgui.new_frame()
        imgui.push_font(self.imgui_font)
        
        self._process_mouse_drag()
        
        self._draw_colorbar()
        
        
        imgui.set_next_window_position(self.window_width - 300, 0)
        imgui.set_next_window_size(300, self.window_height)
        expanded, open = imgui.begin("Info", closable = False, flags = imgui.WINDOW_NO_SCROLLBAR)
        if expanded:
            self._draw_legend()
            self._draw_comparison()
        imgui.end()
        
        if self.results[-1].g_interp is not None:
            imgui.set_next_window_position(self.window_width/2 - 200,0)
            imgui.set_next_window_size(400, 300)
            expanded, open = imgui.begin("Vehicle Report", closable = False, flags = imgui.WINDOW_NO_SCROLLBAR)
            
            if expanded:
                self._draw_vehicle_report()
            imgui.end()
        
        
        imgui.pop_font()
        
        self._update_vehicles()
        
        return
    
    
    def _draw_colorbar(self):
        w = 50
        w_tot = w + 55
        top_pad = 50
        left_pad = 30
        h = self.window_height - top_pad*2
        
        draw_list = imgui.get_overlay_draw_list()
        C_array = get_cmap_rgba(np.flip(np.arange(h)), 0, h)
        
        for (C,py) in zip(C_array, range(top_pad, top_pad+h)):
            y = py / h
            color = imgui.get_color_u32_rgba(*C)
            draw_list.add_line(left_pad, py, left_pad + w, py, color, thickness=1) 
        
        n = 8
        black = imgui.get_color_u32_rgba(0,0,0,1)
        white = imgui.get_color_u32_rgba(1,1,1,1)
        
        draw_list.add_rect_filled(left_pad, top_pad-16, left_pad + 160, top_pad, white)
        draw_list.add_text(left_pad +3, top_pad-16, black, 'Vehicle Speed (m/s)')
        for v, py in zip(np.linspace(self.v_max, self.v_min, n), np.linspace(top_pad, top_pad+h, n)):
            draw_list.add_line(left_pad, py, left_pad + w_tot, py, black, thickness=1) 
            y_offset = -16 if v == self.v_min else 3
            draw_list.add_rect_filled(left_pad + w +3, py + y_offset, left_pad + w_tot, py + y_offset + 16, white)
            draw_list.add_text(left_pad + w +3, py + y_offset, black, '%6.2f'%v)
        
        return
        
    def _draw_legend(self):
        dy = 20
        pad = 10
        draw_list = imgui.get_window_draw_list()
        
        white = imgui.get_color_u32_rgba(1,1,1,1)
        
        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()
        
        px = org.x + off.x
        py = org.y + off.y
        
        for color, label, time in zip(self.colors, self.labels, self.times):
            draw_list.add_rect_filled(px, py+3 , px+30, py+15, color)
            if time is not None:
                draw_list.add_text(px+35, py, white, '%s (%0.2fs)'%(label, time))
            else:
                draw_list.add_text(px+35, py, white, '%s'%(label))
            py += dy
        
        imgui.set_cursor_pos((px, py + dy))
        return
    
    def _draw_comparison(self):
        imgui.columns(3)
        if imgui.radio_button("Play", self.running):
            self.running = not self.running
        imgui.next_column()
        if imgui.radio_button("Reverse", self.reverse):
            self.reverse = not self.reverse
        imgui.next_column()
        if imgui.radio_button("Follow", self.camera_follow):
            self.camera_follow = not self.camera_follow
            
        imgui.columns(1)
        
        
        if self.running and self.animation_t0 is not None:
            dt = (time.time() - self.animation_t0) * self.animation_dt
            self.animation_t += (-dt if self.reverse else dt)
            
            if self.closed:
                if self.animation_t < self.t_min: self.animation_t = self.t_max
                if self.animation_t > self.t_max: self.animation_t = self.t_min
            else:
                if self.animation_t < self.t_min - 2: self.animation_t = self.t_max + 2
                if self.animation_t > self.t_max + 2: self.animation_t = self.t_min - 2
        self.animation_t0 = time.time()
            
        _, self.animation_t  = imgui.slider_float('Time',self.animation_t, min_value = self.t_min, max_value = self.t_max)
        _, self.animation_dt = imgui.slider_float('Speed',self.animation_dt, min_value = 0.1, max_value = 10, flags=imgui.SLIDER_FLAGS_LOGARITHMIC)
        
        vbar = np.clip((self.animation_t - self.t_min)/(self.t_max - self.t_min), 0, 1)
        
        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()
        
        x0 = org.x + off.x
        y0 = org.y + off.y
        plot_size = (0, int((self.window_height - y0)/6) - 4)
        
        plot_multiline(self.t, self.s, self.colors, self.labels, 'Path Length', 
                       t_min = self.t_min, t_max = self.t_max, size = plot_size, vbar = vbar)
        plot_multiline(self.t, self.v, self.colors, self.labels, 'Speed', 
                       t_min = self.t_min, t_max = self.t_max, y_min = self.v_min, y_max = self.v_max, size = plot_size, vbar = vbar)
        plot_multiline(self.t, self.y, self.colors, self.labels, 'Lane Offset', 
                       t_min = self.t_min, t_max = self.t_max, y_min = self.surf.y_min(), y_max = self.surf.y_max(), size = plot_size, vbar = vbar)
        plot_multiline(self.t, self.ths, self.colors, self.labels, 'Heading', 
                       t_min = self.t_min, t_max = self.t_max, size = plot_size, vbar = vbar)
        plot_multiline(self.t, self.uy, self.colors, self.labels, 'Steering Angle', 
                       t_min = self.t_min, t_max = self.t_max, size = plot_size, vbar = vbar)
        plot_multiline(self.t, self.ua, self.colors, self.labels, 'Throttle Command', 
                       t_min = self.t_min, t_max = self.t_max, size = plot_size, vbar = vbar)

    def _update_vehicles(self):

        state = VehicleState()
        for car, result in zip(self.cars, self.results):
            car_t = np.clip(self.animation_t, 0, result.time)
            z = result.z_interp(car_t)
            state.p.s = z[0]
            state.p.y = z[1]
            state.p.ths = z[2]
            state.p.n = self.model.vehicle_config.h
                    
            self.surf.l2gx(state)
            self.surf.l2gq(state)
                    
            car.update_pose(state.x, state.q)
            if self.circular_buf is not None:
                car.circular_buffer_object.update_pose(state.x, state.q)
                        
            if self.camera_follow and result == self.results[-1]:
                self.update_camera_pose(state.x, state.q, zoom = self.model.L*12)
    
    def _draw_vehicle_report(self):
        self.model.vehicle_config
        
        if self.results[-1].g_interp is not None:
            car_t = np.clip(self.animation_t, 0, self.results[-1].time)
            Nf, Nr, Delta = self.results[-1].g_interp(car_t)
            uy = self.results[-1].u_interp(self.animation_t)[-1]
            z  = self.results[-1].z_interp(self.animation_t)
            v = z[3] if len(z) == 4 else np.sqrt(z[3]**2 + z[4]**2)
            
            # get normal forces in kN
            Nfr = (Nf/2 - self.model.tf * Delta) / 1000
            Nfl = (Nf/2 + self.model.tf * Delta) / 1000
            Nrr = (Nr/2 - self.model.tr * Delta) / 1000
            Nrl = (Nr/2 + self.model.tr * Delta) / 1000
            
            N_max = self.model.vehicle_config.N_max / 2 / 1000
            N = [Nfr, Nfl, Nrr, Nrl]
            
            # get tire steering angles
            yfr = np.arctan(np.tan(uy) / (-self.model.tf/2/self.model.L * np.tan(uy) + 1))
            yfl = np.arctan(np.tan(uy) / ( self.model.tf/2/self.model.L * np.tan(uy) + 1))
        
            white = imgui.get_color_u32_rgba(1,1,1,1)
            white_transparent = imgui.get_color_u32_rgba(1,1,1,0.3)
            yellow = imgui.get_color_u32_rgba(1,1,0,1)
            yellow_transparent = imgui.get_color_u32_rgba(1,1,0,.3)
            blue = imgui.get_color_u32_rgba(0,0,1,1)
            
            draw_list = imgui.get_window_draw_list()
            org = imgui.core.get_window_position()
            off = imgui.core.get_cursor_pos()
            
            x0 = org.x + off.x + 60
            y0 = org.y + off.y + 120
            
            thickness = 2
            mark_thickness = 1.5
            
            tf = 40
            tr = 45
            lf = 80
            lr = 90
            
            ph = 20
            pw = 120
            
            box_w = 150
            tire_w = 20
            tire_l = 50
            
            draw_list.add_line(x0 - tf, y0 - lf, x0 + tf, y0 - lf, white_transparent, thickness = thickness)
            draw_list.add_line(x0 - tr, y0 + lr, x0 + tr, y0 + lr, white_transparent, thickness = thickness)
            draw_list.add_line(x0,      y0 - lf, x0,      y0 + lr, white_transparent, thickness = thickness)
            
            draw_list.add_circle(x0 - tf, y0 - lf, 5, yellow, thickness = thickness)
            draw_list.add_circle(x0 + tf, y0 - lf, 5, yellow, thickness = thickness)
            draw_list.add_circle(x0 - tr, y0 + lr, 5, yellow, thickness = thickness)
            draw_list.add_circle(x0 + tr, y0 + lr, 5, yellow, thickness = thickness)
            
            draw_list.add_line(x0 - tf, y0 - lf, x0 - tf + lf - ph/2,   y0 - ph/2,   yellow_transparent, thickness = mark_thickness)
            draw_list.add_line(x0 + tf, y0 - lf, x0 + tf + lf - ph*3/2, y0 - ph*3/2, yellow_transparent, thickness = mark_thickness)
            draw_list.add_line(x0 - tr, y0 + lr, x0 - tr + lr - ph/2,   y0 + ph/2,   yellow_transparent, thickness = mark_thickness)
            draw_list.add_line(x0 + tr, y0 + lr, x0 + tr + lr - ph*3/2, y0 + ph*3/2, yellow_transparent, thickness = mark_thickness)
            
            draw_list.add_line(x0 - tf + lf - ph/2,   y0 - ph/2,   x0 + pw, y0 - ph/2,   yellow_transparent, thickness = mark_thickness)
            draw_list.add_line(x0 + tf + lf - ph*3/2, y0 - ph*3/2, x0 + pw, y0 - ph*3/2, yellow_transparent, thickness = mark_thickness)
            draw_list.add_line(x0 - tr + lr - ph/2,   y0 + ph/2,   x0 + pw, y0 + ph/2,   yellow_transparent, thickness = mark_thickness)
            draw_list.add_line(x0 + tr + lr - ph*3/2, y0 + ph*3/2, x0 + pw, y0 + ph*3/2, yellow_transparent, thickness = mark_thickness)
            
            draw_list.add_line(x0 + pw + 0,         y0 - ph*3/2, x0 + pw + 0,         y0 - ph*2.5, yellow, thickness = 1)
            draw_list.add_line(x0 + pw + box_w - 1, y0 - ph*3/2, x0 + pw + box_w - 1, y0 - ph*2.5, yellow, thickness = 1)
            
            pN = [n / N_max * box_w for n in N]
            draw_list.add_rect_filled(x0 + pw, y0 - ph*2 + 1, x0 + pw + pN[0], y0 - ph   - 1, blue, rounding = 5)
            draw_list.add_rect_filled(x0 + pw, y0 - ph   + 1, x0 + pw + pN[1], y0 -        1, blue, rounding = 5)
            draw_list.add_rect_filled(x0 + pw, y0 + ph*0 + 1, x0 + pw + pN[3], y0 + ph   - 1, blue, rounding = 5)
            draw_list.add_rect_filled(x0 + pw, y0 + ph   + 1, x0 + pw + pN[2], y0 + ph*2 - 1, blue, rounding = 5)
            
            draw_list.add_rect(x0 + pw, y0 - ph*2 + 1, x0 + pw + box_w, y0 - ph   - 1, yellow, rounding = 5, thickness = thickness)
            draw_list.add_rect(x0 + pw, y0 - ph   + 1, x0 + pw + box_w, y0 -        1, yellow, rounding = 5, thickness = thickness)
            draw_list.add_rect(x0 + pw, y0 + ph*0 + 1, x0 + pw + box_w, y0 + ph   - 1, yellow, rounding = 5, thickness = thickness)
            draw_list.add_rect(x0 + pw, y0 + ph   + 1, x0 + pw + box_w, y0 + ph*2 - 1, yellow, rounding = 5, thickness = thickness)
            
            draw_list.add_text(x0 + pw + 2,         y0 - ph * 3, yellow, '0')
            draw_list.add_text(x0 + pw + box_w + 2, y0 - ph * 3, yellow, '%dkN'%N_max)
            
            draw_list.add_rect(x0 - tr - tire_w/2, y0 + lr - tire_l/2, x0 - tr + tire_w/2, y0 + lr + tire_l/2, white, thickness = thickness)
            draw_list.add_rect(x0 + tr - tire_w/2, y0 + lr - tire_l/2, x0 + tr + tire_w/2, y0 + lr + tire_l/2, white, thickness = thickness)
            
            xfr = np.array([[-tire_w/2, -tire_l/2], 
                            [ tire_w/2, -tire_l/2],
                            [ tire_w/2,  tire_l/2],
                            [-tire_w/2,  tire_l/2]]) @ np.array([[np.cos(yfr), -np.sin(yfr)],[np.sin(yfr), np.cos(yfr)]]) + np.array([[x0 + tf, y0 - lf]])
            xfl = np.array([[-tire_w/2, -tire_l/2], 
                            [ tire_w/2, -tire_l/2],
                            [ tire_w/2,  tire_l/2],
                            [-tire_w/2,  tire_l/2]]) @ np.array([[np.cos(yfl), -np.sin(yfl)],[np.sin(yfl), np.cos(yfl)]]) + np.array([[x0 - tf, y0 - lf]])
            
            draw_list.add_polyline(xfr.tolist(), white, thickness = thickness)
            draw_list.add_polyline(xfl.tolist(), white, thickness = thickness)
            
            imgui.push_font(self.big_imgui_font)
            draw_list.add_text(x0 + 120, y0 + 80, white, '%0.2fm/s'%v)
            imgui.pop_font()

