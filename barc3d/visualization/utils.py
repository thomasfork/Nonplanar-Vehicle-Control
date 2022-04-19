import os
import numpy as np
import imgui 
from stl import mesh

from barc3d.visualization.shaders import vtype

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

COLORMAP_NAME = 'plasma'
CMAP = plt.get_cmap(COLORMAP_NAME)
CMAP_NORM = mpl.colors.Normalize(vmin=0, vmax=1)
CMAP_SCALAR = cm.ScalarMappable(norm = CMAP_NORM, cmap = CMAP)

def get_cmap_rgba(v, v_min, v_max):
    v_rel = (v - v_min) / (v_max - v_min)
    return CMAP_SCALAR.to_rgba(v_rel)
    

def load_ford_mustang(model, color = [1,0,0,1]):
    folder = os.path.dirname(__file__)
    filename = os.path.join(folder, 'textures','mustang.stl')
    
    stl_mesh = mesh.Mesh.from_file(filename)
    
    Vx = stl_mesh.vectors.reshape([-1,3])
    Vn = np.repeat(stl_mesh.normals,3,axis = 0)
    
    Vx = Vx[:,[1,0,2]] # raw data axes are flipped
    Vn = Vn[:,[1,0,2]]
    
    # not all STL files give normal vectors of unit length
    Vn_norm = np.linalg.norm(Vn, axis = 1)
    Vn_norm[Vn_norm == 0] = 1
    Vn = (Vn.T / Vn_norm).T 
    
    Vx[:,0] *= -1
    
    xl = np.min(Vx,axis = 0)
    xu = np.max(Vx,axis = 0)
    
    box = xu-xl
    
    scale = (model.bf + model.br) / box[0] 
    
    
    Vx *= scale
    
    Vx[:,0] = Vx[:,0] - np.min(Vx[:,0]) - model.br 
    
    Vx[:,2] = Vx[:,2] - np.min(Vx[:,2]) - model.h
    
    V = np.zeros(len(Vx), dtype=vtype)
    V["a_position"] = Vx
    V['a_color'] = color
    V['a_normal'] = Vn
    I = np.arange(len(Vx))
    I = I.astype(np.uint32)
    
    
    return V, I

def load_trajectory(surf, states, w = 0.4, h = 0.05, road_offset =-0.592, v_max = None, v_min = None, closed = False):

    x = np.array([surf.ro2x((state.p.s, state.p.y, state.p.ths), road_offset) for state in states]).squeeze()
    v  = np.array([state.v.mag() for state in states])
    e2 = np.array([state.q.e2() for state in states]).squeeze()
    e3 = np.array([state.q.e3() for state in states]).squeeze()
    et = np.gradient(x, axis = 1)
    
    if v_max is None:
        v_max = v.max()
    if v_min is None:
        v_min = v.min()
    
    V = np.concatenate([x + e2*w + e3*h, 
                        x + e2*w,
                        x - e2*w, 
                        x - e2*w + e3*h])
    
    C = get_cmap_rgba(v, v_min, v_max)
       
    #C = (v - v_min) / (v_max - v_min)
    #C = np.array([np.ones(C.shape), 1-C, np.zeros(C.shape), np.ones(C.shape)]).T
    C = np.tile(C.T, 4).T
    
    N = np.zeros(V.shape)
    N[:,-1] = 1
    
    n = len(states)
    # first face
    I = np.array([[0,1,n],
                  [1,n+1,n]]) 
    # first segment
    I = np.concatenate([I, I+n, I+2*n, I+3*n])
    I = I % (n*4)
    #all segments
    I = np.concatenate([I + k for k in range(n-1)])    
    
    if not closed:
        # cap the ends
        I = np.concatenate([I, np.array([[0,n,2*n],
                                         [0,2*n, 3*n]])])
        I = np.concatenate([I, n-1+ np.array([[0,2*n,n],
                                     [0,3*n, 2*n]])])
    else:
        # join start and end
        I_join =  np.array([[0,n,n-1], [n,2*n-1,n-1]])
        I_join = np.concatenate([I_join, I_join+n, I_join+2*n, I_join+3*n]) % (4*n)
        
        I = np.concatenate([I, I_join])
        
    Vertices = np.zeros(V.shape[0], dtype=vtype)
    Vertices['a_position'] = V 
    Vertices['a_color']    = C 
    Vertices['a_normal']   = N
    
    return Vertices, np.concatenate(I).astype(np.uint32) 
    
def load_circle(n = 40, ri = None, ro = 2, t = 0.2, C = np.array([0,0,1,1]), h = 0):
    if ri is None:
        ri = max(ro - 0.5, 0.1)
        
    th = np.linspace(0, 2*np.pi, n)
    
    er = np.array([np.cos(th), np.sin(th), np.zeros(th.shape)]).T
    en = np.array([0,0,1])
    
    V = np.concatenate([er*ri - en*h, er*ro - en*h, er*ro + en*(t-h), er*ri + en*(t-h)])
    
    # first face
    I = np.array([[0,1,n],
                  [1,n+1,n]]) 
    # first segment
    I = np.concatenate([I, I+n, I+2*n, I+3*n])
    I = I % (n*4)
    #all segments
    I = np.concatenate([I + k for k in range(n-1)])  
    
    N = np.zeros(V.shape)
    N[:,-1] = 1
    
    Vertices = np.zeros(V.shape[0], dtype=vtype)
    Vertices['a_position'] = V 
    Vertices['a_color']    = C 
    Vertices['a_normal']   = N
    
    return Vertices, np.concatenate(I).astype(np.uint32) 

    
PLOT_BACKGROUND_COLOR = [0.13725491, 0.20784314, 0.30980393, 0.7254902 ]
PLOT_BACKGROUND_COLOR = [0,0,0,0.7]
def plot_multiline(time, data, colors, labels, title, t_min = None, t_max = None, y_min = None, y_max = None, size = (0,150), vbar = None):
    if t_min is None: t_min = time.min()
    if t_max is None: t_max = time.max()
    if y_min is None: y_min = data.min()
    if y_max is None: y_max = data.max()
    
    if size[0] <= 0: size = (imgui.core.get_window_content_region_width(), size[1]) 
    draw_list = imgui.get_window_draw_list()
        
    org = imgui.core.get_window_position()
    off = imgui.core.get_cursor_pos()
    
    x0 = org.x + off.x
    y0 = org.y + off.y
    
    white = imgui.get_color_u32_rgba(1,1,1,1)
    
    draw_list.add_rect_filled(x0, y0, x0+size[0], y0+size[1], imgui.get_color_u32_rgba(*PLOT_BACKGROUND_COLOR), 5)
    
    for t, line, color in zip(time, data, colors):
    
        px = (t - t_min) / (t_max - t_min) * size[0] + x0 
        py = (line - y_min) / (y_max - y_min)
        py = y0 + 1 + (size[1]-2) * (1 - py)
        
        draw_list.add_polyline(np.array([px,py]).T.tolist(), color, closed=False)
    
    
    imgui.begin_child(title, size[0], size[1], border=True)
    imgui.end_child()
    
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        
        org = imgui.core.get_window_position()
        off = imgui.core.get_cursor_pos()
        tooltip_draw_list = imgui.get_window_draw_list()
        px = org.x + off.x+5 
        py = org.y + off.y+2
        dy = 20
        
        mouse_pos = imgui.core.get_mouse_pos()
        t_tooltip = t_min + (t_max - t_min) * (mouse_pos.x - x0) / size[0]
        
        no_tooltip_lines = 0
        for t, line, color, label in zip(time, data, colors, labels):
            if t_tooltip < t[0]:
                continue
            idx = np.searchsorted(t, t_tooltip)
            
            if idx < len(line):
                tooltip_draw_list.add_rect_filled(px, py+3 , px+30, py+15, color)
                tooltip_draw_list.add_text(px+35, py, white, '(%9.3f) %s'%(line[idx], label))
                py += dy
                
                no_tooltip_lines += 1
        
        imgui.begin_child("region", 300, dy*no_tooltip_lines, border=False)
        imgui.end_child()
        imgui.end_tooltip()
    if vbar is not None:
        draw_list.add_line(x0 + size[0]*vbar, y0, x0 + size[0]*vbar, y0 + size[1], white, 2)
        
        
    draw_list.add_text(x0, y0, white, '%8.2f'%y_max)
    draw_list.add_text(x0, y0+size[1]-18, white, '%8.2f'%y_min)
    draw_list.add_line(x0, y0,x0+60, y0, white)
    draw_list.add_line(x0, y0+size[1],x0+60, y0+size[1], white)
    
    if title is not None:
        draw_list.add_text(x0+102, y0+2, white, str(title))
  
