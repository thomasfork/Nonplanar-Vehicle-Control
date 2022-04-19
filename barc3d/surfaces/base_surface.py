import numpy as np
from abc import abstractmethod, ABC
import casadi as ca

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.transform import Rotation
import scipy.interpolate


from barc3d.pytypes import ParametricPose, ParametricVelocity, OrientationQuaternion, VehicleState
import barc3d.surfaces as surfaces
from barc3d.visualization.shaders import vtype
INTERP_LINEAR = 0
INTERP_PCHIP  = 1
INTERP_PARTIAL_PCHIP = 2

class BaseSurface(ABC):
    def get_class_label(self):
        return type(self).__name__
        
    def save_surface(self, surface_name):
        filename = surfaces.surface_name_to_filename(surface_name)
        type_str = self.get_class_label()
        np.savez(filename, surf_class = type_str, **vars(self.config))
        print('Generated surface %25s of type %s'%(surface_name, self.get_class_label()))
        return
    
    def is_class_data(self, data):
        if 'surf_class' not in data.keys():
            return False
        if data['surf_class'].__str__() == self.get_class_label():
            return True
        return False
        
    def unpack_loaded_data(self, data):
        for key in vars(self.config).keys():
            if key in data:
                self.config.__setattr__(key, data[key])
            else:
                print('No source data for key %s, likely loaded wrong surf file'%key)
        return
    
    def pchip_interpolant(self, s, var):
        # close to linear interpolation but smoother. 
        k = scipy.interpolate.PchipInterpolator(s,var)
        c = k.c.copy()
        x = k.x.copy()
        
        kn = k(x[-1])
        
        for k in range(len(x)-1):
            c0n = c[3,k] -   c[2,k] * x[k] +   c[1,k] * x[k]**2 - c[0,k] * x[k]**3
            c1n = c[2,k] - 2*c[1,k] * x[k] + 3*c[0,k] * x[k]**2
            c2n = c[1,k] - 3*c[0,k] * x[k]
            
            c[3,k] = c0n
            c[2,k] = c1n
            c[1,k] = c2n
            
        s = ca.SX.sym('s')
        c0 = ca.pw_const(s, x, [c[3,0],*c[3,:],kn])
        c1 = ca.pw_const(s, x, [0,*c[2,:],0])
        c2 = ca.pw_const(s, x, [0,*c[1,:],0])
        c3 = ca.pw_const(s, x, [0,*c[0,:],0])
        
        k = c0 + c1*s + c2*s**2 + c3*s**3
        fk = ca.Function('k', [s], [k])
        
        return fk
        
    def fill_in_sx(self, name, expr, pose, param_terms, pose_mx, f_param_terms):
        '''
        helper function for the tedious process of converting a fast SX expression into an MX function
        this has to be done since interpolation objects in CasADi are MX whereas SX evaluate much faster
        
        expr, pose, and param_terms are all SX
        expr = f(pose,param_terms) = f_MX(pose_mx, f_param_terms(pose_mx)) = g(pose_mx)
        
        the final expression "g" is returned.
        '''
        
        func = ca.Function(name, [pose, param_terms], [expr])
        feval = func.call([pose_mx, f_param_terms(pose_mx)])
        
        return ca.Function(name, [pose_mx], feval)

    @abstractmethod
    def initialize(self, config):
        return
        
    def s_min(self, y = None):
        return self.config.s_min
        
    def s_max(self, y = None):
        return self.config.s_max
    
    def y_min(self, s = None):
        if isinstance(s, np.ndarray):
            return np.ones(s.size) * self.config.y_min
        else:
            return self.config.y_min
    
    def y_max(self, s = None):
        if isinstance(s, np.ndarray):
            return np.ones(s.size) * self.config.y_max
        else:
            return self.config.y_max
    
    def xps(self, pos:ParametricPose):
        return self.pose2xps((pos.s, pos.y, pos.ths))
        
    def xpy(self, pos:ParametricPose):
        return self.pose2xpy((pos.s, pos.y, pos.ths))
        
    def xpn(self, pos:ParametricPose):
        return self.pose2xpn((pos.s, pos.y, pos.ths))
    
    def xpss(self, pos:ParametricPose):
        return self.pose2xpss((pos.s, pos.y, pos.ths))
        
    def xpsy(self, pos:ParametricPose):
        return self.pose2xpsy((pos.s, pos.y, pos.ths))
        
    def xpyy(self, pos:ParametricPose):
        return self.pose2xpyy((pos.s, pos.y, pos.ths))
        
    def I(self, pos:ParametricPose):
        return self.pose2one((pos.s, pos.y, pos.ths))
    
    def Iinv(self, pos: ParametricPose):
        return np.linalg.inv(self.I(pos))
        
    def II(self, pos: ParametricPose):
        return self.pose2two((pos.s, pos.y, pos.ths))
                         
    def ws(self, pos:ParametricPose):
        return self.pose2ws((pos.s, pos.y, pos.ths))
   
    def wy(self, pos:ParametricPose):
        return self.pose2wy((pos.s, pos.y, pos.ths))
        
    def J(self, pos: ParametricPose, q: OrientationQuaternion):
        return self.pose2J((pos.s, pos.y, pos.ths))
    
    def Jinv(self, pos: ParametricPose, q: OrientationQuaternion):
        return np.linalg.inv(self.J(pos, q))
        
    def R(self, pos: ParametricPose, q: OrientationQuaternion):
        return self.pose2R((pos.s, pos.y, pos.ths))
        
    def ro2n(self, pose):
        return self.pose2xpn(pose)
    
    def Wmat(self, pos: ParametricPose, q: OrientationQuaternion):
        return self.Jinv(pos, q) @ self.II(pos) @ self.Iinv(pos) @ self.J(pos, q)
   
    def local_to_global(self, state:VehicleState):
        '''
        converts parametric pose to global pose 
        useful for plotting when pose is an input.
        '''
        self.l2gx(state) # position
        self.l2gq(state) # orientation
        return
    
    def frenet_to_global(self, state:VehicleState):
        '''
        converts states updated by the generalized frenet model to ones not updated
        this means converting
        ths -> orientation quaternion
        body velocity -> parametric velocity and angular velocity
        parametric position -> global position
        ''' 
        self.l2gx(state) # position
        self.l2gq(state) # orientation
        self.b2lv(state) # parametric velocity
        self.bv2bw(state) # angular velocity
        return
    
    def l2gx(self, state:VehicleState):
        '''
        parametric position -> global position
        '''
        
        x = (self.ro2x((state.p.s, state.p.y, state.p.ths), state.p.n)).squeeze()
        state.x.from_vec(x)
    
    def l2gq(self, state:VehicleState):
        ''' converts state.p pose variables into state.q '''
        R = self.pose2R((state.p.s, state.p.y, state.p.ths))
        q = Rotation.from_matrix(R).as_quat()
        state.q.from_vec(q)
        
    def b2lv(self, state: VehicleState):
        '''
        converts body velocity to body parametric velocity
        '''
        ds_dy = self.Iinv(state.p) @ self.J(state.p, state.q) @ np.array([state.v.v1, state.v.v2])
        state.pt.ds = ds_dy[0]
        state.pt.dy = ds_dy[1]
        state.pt.dths = state.w.w3 + \
                             state.pt.ds * self.ws(state.p) + \
                             state.pt.dy * self.wy(state.p)
        return
    
    def bv2bw(self, state:VehicleState):
        '''
        updates w1 and w2 given the current vehicle body velocity
        '''
        w2_w1 = self.Jinv(state.p, state.q) @ self.II(state.p) @ np.array([state.pt.ds,state.pt.dy])
        state.w.w2 = -w2_w1[0]
        state.w.w1 = w2_w1[1]
        return
    
    def g2lx(self, state:VehicleState):
        '''
        global position -> parametric position
        '''
        raise NotImplementedError('')
        return
    
    def g2lq(self, state:VehicleState):
        '''
        global orientation -> parametric orientation
        '''
        raise NotImplementedError('')
        return
        
    def ro2x(self, pose, n = 0):
        return self.pose2xp(pose) + self.ro2n(pose) * n
    
    def ro2n(self, pose):
        return self.pose2xpn(pose)
    
    def parametric_grid(self, n_s = 100, n_y = 30, s_ext = 0):
        '''
        grid the surface into a pair of NxM arrays corresponding to the u,v state of every point on the surface. 
        main use is plotting and Voronoi projection
        '''
        s = np.linspace(self.s_min(0) - s_ext, self.s_max(0) + s_ext, n_s)
        y = np.linspace(self.y_min(0), self.y_max(0), n_y)
        Y,S = np.meshgrid(y,s)
        return S,Y  
        
    def generate_texture(self, S, Y, thickness = None, road_offset = 0, as_opengl = True, y_ext = 0):
        '''
        generates a texture in the form Vertices (V), Faces (I)
        Edges are most likely unecessary.
        
        thickness corresponds to the distance from the top of the road surface to the bottom
        road_offset is meant for offsetting the parametric surface to the actual road geometry 
        
        S,Y are NxM arrays of (s,y) pairs adjacent to one another on the surface and should be obtainable by self.parametric_grid
        '''
        if thickness is None:
            thickness = (self.y_max(0) - self.y_min(0)) / 5
        
        n_s = S.shape[0]
        n_y = Y.shape[1]
        
        assert S.shape[0] == Y.shape[0]
        assert S.shape[1] == Y.shape[1]
        
        # vertex data for upper surface
        V_array1 = np.zeros((n_s*n_y, 3))  # mesh vertices (xi,xj,xk)
        C_array1 = np.zeros((n_s*n_y, 4))  # mesh colors   (r,g,b,a)
        n_array1 = np.zeros((n_s*n_y, 3))  # mesh normal vectors (ni, nj, nk)
        T_array1 = np.zeros((n_s*n_y, 2))  # mesh texture  (s,y)

        # vertex data for lower surface
        V_array2 = V_array1.copy()
        C_array2 = C_array1.copy()
        n_array2 = n_array1.copy()
        T_array2 = T_array1.copy()
        
        # compute vertex properties for top and bottom
        pose = np.array([np.concatenate(S), np.concatenate(Y), np.zeros(np.prod(S.size))])
            
        V_array1 = self.ro2x(pose, road_offset).T
        V_array2 = self.ro2x(pose, road_offset - thickness).T
            
        n_array1 = self.ro2n(pose).T
        n_array2 =-n_array1.copy()
            
        s = pose[0]
        y = pose[1]
        if isinstance(self, BaseCenterlineSurface):
            c = - (y - self.y_max(s))*(y -self.y_min(s)) / (self.y_max(s) - self.y_min(s))**2 * 4 / 2
            c = c.clip(0,1).squeeze()
            
            C_array1 = np.array([c,c,c, np.ones(c.shape)]).T
            C_array2 = np.concatenate([np.zeros((c.shape[0], 3)), np.ones((c.shape[0],1))], axis = 1)
        else:
            # checkerboard
            # c = (((s + s.min()) / 5).astype(np.int) + ((y + y.min()) / 5).astype(np.int)) % 2
            # smooth checkerboard
            c = np.cos(s * 2 * np.pi / 30) * np.cos(y * 2 * np.pi / 30)
            c = c - c.min()
            c = c / c.max()
            
            
            C_array1 = np.array([c,c,c, np.ones(c.shape)]).T
            C_array2 = np.concatenate([np.zeros((c.shape[0], 3)), np.ones((c.shape[0],1))], axis = 1)
          
        T_array1 = np.zeros((c.shape[0],2))
        T_array2 = np.zeros((c.shape[0],2))
            
        # now compute indices that correspond to triangles and edges for upper surface
        I_array = np.zeros((n_s-1, n_y-1, 2,3))
        O_array = np.zeros((n_s-1, n_y-1, 4,2))
        
        idxs = np.arange(len(V_array1)).reshape(n_s,n_y)
        
        I_array[:,:,0,0] = idxs[:-1,:-1]       # bot left
        I_array[:,:,0,1] = idxs[1: ,:-1]       # bot right
        I_array[:,:,0,2] = idxs[:-1,1: ]       # top left
        I_array[:,:,1,0] = idxs[1: ,1: ]       # top right
        I_array[:,:,1,1] = idxs[:-1,1: ]       # top left
        I_array[:,:,1,2] = idxs[1: ,:-1]       # bot right
        
        # merge lower and upper surfaces
        offset = V_array1.shape[0]                     #index offset
        
        # copy for lower indices and reverse triangle order for face culling
        I_array_lower = I_array + offset
        I_array_lower[:,:,:,[0,1,2]] = I_array_lower[:,:,:,[0,2,1]]

        I_array_lower = I_array_lower.reshape(-1,3)
        I_array = I_array.reshape(-1,3)
        
        
        #combine the surfaces
        V_tot = np.vstack([V_array1, V_array2])
        C_tot = np.vstack([C_array1, C_array2])
        n_tot = np.vstack([n_array1, n_array2])
        T_tot = np.vstack([T_array1, T_array2])
        I_tot = np.vstack([I_array, I_array_lower])
        
        idxs_lower = idxs + offset
        
        # add triangles along the sides of the track
        I_s_border = np.zeros((n_s-1, 4, 3))
        I_y_border = np.zeros((n_y-1, 4, 3))
        
        I_s_border[:, 0, 0] = idxs[:-1, 0]           # right  top start
        I_s_border[:, 0, 1] = idxs[:-1, 0] + offset  # right  bot start
        I_s_border[:, 0, 2] = idxs[1:,  0]           # right  top end
        I_s_border[:, 1, 0] = idxs[1:,  0] + offset  # right  bot end
        I_s_border[:, 1, 2] = idxs[:-1, 0] + offset  # right  bot start
        I_s_border[:, 1, 1] = idxs[1:,  0]           # right  top end

        I_s_border[:, 2, 0] = idxs[:-1,-1]           # left top start
        I_s_border[:, 2, 2] = idxs[:-1,-1] + offset  # left bot start
        I_s_border[:, 2, 1] = idxs[1:, -1]           # left top end
        I_s_border[:, 3, 0] = idxs[1:, -1] + offset  # left bot end
        I_s_border[:, 3, 1] = idxs[:-1,-1] + offset  # left bot start
        I_s_border[:, 3, 2] = idxs[1:, -1]           # left top end
        
        I_y_border[:, 0, 0] = idxs[0, :-1]           # left  top start
        I_y_border[:, 0, 2] = idxs[0, :-1] + offset  # left  bot start
        I_y_border[:, 0, 1] = idxs[0, 1: ]           # left  top end
        I_y_border[:, 1, 0] = idxs[0, 1: ] + offset  # left  bot end
        I_y_border[:, 1, 1] = idxs[0, :-1] + offset  # left  bot start
        I_y_border[:, 1, 2] = idxs[0, 1: ]           # left  top end
        I_y_border[:, 2, 0] = idxs[-1,:-1]           # right top start
        I_y_border[:, 2 ,1] = idxs[-1,:-1] + offset  # right bot start
        I_y_border[:, 2 ,2] = idxs[-1,1: ]           # right top end
        I_y_border[:, 3, 0] = idxs[-1,1: ] + offset  # right bot end
        I_y_border[:, 3, 2] = idxs[-1,:-1] + offset  # right bot start
        I_y_border[:, 3, 1] = idxs[-1,1: ]           # right top end
        
        I_tot = np.vstack([I_tot, I_s_border.reshape(-1,3), I_y_border.reshape(-1,3)])
        
        if as_opengl:
            I_tot = np.concatenate(I_tot)
             
            V_opengl = np.zeros(V_tot.shape[0], dtype=vtype)
            V_opengl['a_position'] = V_tot 
            V_opengl['a_color']    = C_tot 
            V_opengl['a_normal']   = n_tot 
            
            if y_ext > 0:
                V, I = self.generate_siderails(s = S[:,0], thickness = thickness, road_offset = road_offset, y_ext = y_ext)
                
                V_merged = np.zeros(V_tot.shape[0] + V.shape[0], dtype=vtype)
                V_merged['a_position'] = np.concatenate([V_opengl['a_position'], V['a_position']])
                V_merged['a_color']    = np.concatenate([V_opengl['a_color'], V['a_color']])
                V_merged['a_normal']   = np.concatenate([V_opengl['a_normal'], V['a_normal']])
                
                I_merged = np.concatenate([I_tot, I + len(V_opengl)])
                return V_merged, I_merged.astype(np.uint32)
                
            return V_opengl, I_tot.astype(np.uint32)
        else:
            return V_tot, I_tot 
    
    def generate_siderails(self, s = None, thickness = None, road_offset = 0, y_ext = 0):
        if y_ext <= 0:
            return
        if thickness is None:
            thickness = (self.y_max(0) - self.y_min(0)) / 5
        if s is None:
            s = np.linspace(self.s_min(), self.s_max(), 1000)
        n = len(s)
        V, I = [], []
        for y in [self.y_min(), self.y_max()]:
            S = s
            Y = np.ones(S.shape) * y
            
            pose = np.array([S, Y, np.zeros(np.prod(S.size))])
            ey_array = self.pose2xpy(pose)
            ey_array = ey_array / np.linalg.norm(ey_array, axis = 0)

            V_array1 = self.ro2x(pose, road_offset) + ey_array * y_ext * (1 if y == self.y_max() else 0)
            V_array2 = self.ro2x(pose, road_offset - thickness) + ey_array * y_ext * (1 if y == self.y_max() else 0)
            
            n_array1 = self.ro2n(pose)
            n_array2 =-n_array1.copy()
            n_array2 = n_array1.copy() / 2
        
            V_array3 = V_array1 + ey_array * y_ext * (-1)
            V_array4 = V_array2 + ey_array * y_ext * (-1)
            
            V_tot = np.concatenate([V_array1, V_array2, V_array4, V_array3], axis = 1)
            n_tot = np.concatenate([n_array1, n_array2, n_array2, n_array1], axis = 1)
            
            I_array1 = np.array([np.arange(n-1), np.arange(1,n), np.arange(1,n)+n, np.arange(n-1), np.arange(1,n)+n, np.arange(n-1)+n]).T
            I_array1 = np.concatenate(I_array1)
            I_array_start_cap = np.array([0,n,2*n, 0,2*n,3*n])
            I_array_end_cap =   np.array([0,2*n,n, 0,3*n,2*n])+ (n-1)
            
            I_tot = np.concatenate([I_array1, I_array1 + n, I_array1 + 2*n, I_array1 + 3*n, I_array_start_cap, I_array_end_cap])
            I_tot = np.mod(I_tot, 4*n)
            C_tot = np.array([0.8,0.8,0.8,1])
                 
            V_opengl = np.zeros(V_tot.shape[1], dtype=vtype)
            V_opengl['a_position'] = V_tot.T
            V_opengl['a_color']    = C_tot 
            V_opengl['a_normal']   = n_tot.T
            V.append(V_opengl)
            I.append(I_tot)
        
        V = np.concatenate(V)
        I = np.concatenate([I[0], I[1] + 4*n])
        
        return V, I.astype(np.uint32)
        
    
    def preview_surface(self, block = True):
        S,Y = self.parametric_grid()
        V,I = self.generate_texture(S,Y, as_opengl = False)
        x = V[:,0]
        y = V[:,1]
        z = V[:,2]
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_trisurf(x, y, z, triangles=I)
        plt.show(block = block)
    
        return

    def _post_sym_rep(self):
        '''
        take partial derivatives of surface and compute remaining terms
        this is the same for any surface, hence has its own function
        '''
        xps = self.sym_rep['xps']
        xpy = self.sym_rep['xpy']
        
        xpn = ca.cross(xps, xpy)
        xpn = xpn / ca.sqrt(xpn.T @ xpn)
        self.sym_rep['xpn'] = xpn
        
        xpss = self.sym_rep['xpss']
        xpsy = self.sym_rep['xpsy']
        xpys = self.sym_rep['xpys']
        xpyy = self.sym_rep['xpyy']
        
        ths = self.sym_rep['ths']
        
        # terms computed from parametric surface
        one = ca.vertcat(ca.horzcat(xps.T @ xps, xps.T @ xpy),\
                         ca.horzcat(xpy.T @ xps, xpy.T @ xpy)) 
        
        two = ca.vertcat(ca.horzcat(xpss.T @ xpn, xpsy.T @ xpn),
                         ca.horzcat(xpys.T @ xpn, xpyy.T @ xpn))
                   
        
        ws = (ca.cross(xpss, xps).T @ xpn) / (xps.T @ xps)
        wy = (ca.cross(xpsy, xps).T @ xpn) / (xps.T @ xps)
        
        
        # angle between partial derivatives and jacobian 
        thp = - ca.arcsin(xps.T @ xpy / ca.sqrt(xps.T @ xps) / ca.sqrt(xpy.T @ xpy))
        J = ca.vertcat(ca.horzcat(ca.cos(ths) * ca.sqrt(xps.T @ xps),-ca.sin(ths) * ca.sqrt(xps.T @ xps)),
                       ca.horzcat(ca.sin(ths - thp) * ca.sqrt(xpy.T @ xpy), ca.cos(ths - thp) * ca.sqrt(xpy.T @ xpy)))
        
        # rotation matrix of vehicle
        R = ca.horzcat(ca.horzcat(xps, xpy) @ (ca.inv(one)  @ J), xpn)
        
        one = ca.simplify(one)
        two = ca.simplify(two)
        ws  = ca.simplify(ws)
        wy  = ca.simplify(wy)
        J   = ca.simplify(J)
        R   = ca.simplify(R)
        
        self.sym_rep['one'] = one
        self.sym_rep['two'] = two
        self.sym_rep['ws'] = ws
        self.sym_rep['wy'] = wy
        self.sym_rep['thp'] = thp
        self.sym_rep['J'] = J
        self.sym_rep['R'] = R
        

    def _unpack_sym_rep(self):
        '''
        unpack the major parts of the surface representation and create functions for them for evaluation, ie. converting parametric state to global state
        '''
        pose = self.sym_rep['pose']
        pose_mx = self.sym_rep['pose_mx']
        param_terms = self.sym_rep['param_terms']
        f_param_terms = self.sym_rep['f_param_terms']
        
        xps = self.sym_rep['xps']
        xpy = self.sym_rep['xpy']
        xpn = self.sym_rep['xpn']
        
        xpss = self.sym_rep['xpss']
        xpsy = self.sym_rep['xpsy']
        xpys = self.sym_rep['xpys']
        xpyy = self.sym_rep['xpyy']
        
        one = self.sym_rep['one']
        two = self.sym_rep['two']
        ws = self.sym_rep['ws']
        wy = self.sym_rep['wy']
        J = self.sym_rep['J']
        R = self.sym_rep['R']
        
        # create functions to evaluate the above for current surface 
        f_xps  = self.fill_in_sx('xps',  xps,  pose, param_terms, pose_mx, f_param_terms)
        f_xpy  = self.fill_in_sx('xpy',  xpy,  pose, param_terms, pose_mx, f_param_terms)
        f_xpn  = self.fill_in_sx('xpn',  xpn,  pose, param_terms, pose_mx, f_param_terms)
        
        f_xpss = self.fill_in_sx('xpss', xpss, pose, param_terms, pose_mx, f_param_terms)
        f_xpsy = self.fill_in_sx('xpsy', xpsy, pose, param_terms, pose_mx, f_param_terms)
        f_xpys = self.fill_in_sx('xpys', xpys, pose, param_terms, pose_mx, f_param_terms)
        f_xpyy = self.fill_in_sx('xpyy', xpyy, pose, param_terms, pose_mx, f_param_terms)
        
        f_one = self.fill_in_sx('one', one, pose, param_terms, pose_mx, f_param_terms)
        f_two = self.fill_in_sx('two', two, pose, param_terms, pose_mx, f_param_terms)
        f_ws  = self.fill_in_sx('ws',  ws,  pose, param_terms, pose_mx, f_param_terms)
        f_wy  = self.fill_in_sx('wy',  wy,  pose, param_terms, pose_mx, f_param_terms)
        f_J   = self.fill_in_sx('J',   J,   pose, param_terms, pose_mx, f_param_terms)
        f_R   = self.fill_in_sx('R',   R,   pose, param_terms, pose_mx, f_param_terms)
        
        
        # package in converter functions that convert to numpy
        self.pose2xps = lambda pose: np.asarray(f_xps(pose))
        self.pose2xpy = lambda pose: np.asarray(f_xpy(pose))
        self.pose2xpn = lambda pose: np.asarray(f_xpn(pose))
        
        self.pose2xpss = lambda pose: np.asarray(f_xpss(pose))
        self.pose2xpsy = lambda pose: np.asarray(f_xpsy(pose))
        self.pose2xpys = lambda pose: np.asarray(f_xpys(pose))
        self.pose2xpyy = lambda pose: np.asarray(f_xpyy(pose))
        
        self.pose2one = lambda pose: np.asarray(f_one(pose))
        self.pose2two = lambda pose: np.asarray(f_two(pose))
        self.pose2ws  = lambda pose: np.asarray(f_ws(pose))
        self.pose2wy  = lambda pose: np.asarray(f_wy(pose))
        self.pose2J   = lambda pose: np.asarray(f_J(pose))      
        self.pose2R   = lambda pose: np.asarray(f_R(pose))   
        
        # store some specific function in sym rep, ie. for OBCA
        # xp must be provided by the specific surface, ie. centerline, elevation, etc...
        self.sym_rep['f_R'] = f_R
        self.sym_rep['f_xpn'] = f_xpn
    
class BaseCenterlineSurface(BaseSurface):
    
    '''
    class of surfaces that reference a centerline, ie. the traditional Frenet Frame (surface)
    
    These may or may not be cross-sectionally linear, for those use "BaseLinearCenterlineSurface"
    '''
    def initialize(self, config):
        self.config = config
        
        self._setup_interp()
        self._compute_sym_rep()
        self._post_sym_rep()
        self._generate_centerline()
        self._post_centerline()
        self._unpack_sym_rep() 
        
        self.initialized = True
        return
    
    @abstractmethod
    def _setup_interp(self):
        '''
        set up interpolation objects for the surface, ie. curvature as a function of path length
        exact variables depend on the class of surface
        '''
        return
    
    @abstractmethod
    def _compute_sym_rep(self):
        '''
        compute symbolic represnetation of the surface
        must include partial derivatives as a function of surface parameterization
        ie. track heading and derivative thereof -> partial derivative of resulting surface
        see existing surfaces for how this is stored in self.sym_rep
        '''
        return
        
    def _generate_centerline(self):
        ''' 
        for centerline surfaces treated here the centerline is implicit, and must be found via integration
        this does so and sets up xc - the centerline position, as an interpolation object.
        '''
        s = self.sym_rep['s_mx']       
        x = ca.MX.sym('x',3)       
        
        pose = self.sym_rep['pose']
        pose_mx = self.sym_rep['pose_mx']
        param_terms = self.sym_rep['param_terms']
        f_param_terms = self.sym_rep['f_param_terms']
        es = self.sym_rep['es']
        
        fes = self.fill_in_sx('es', es, pose, param_terms, pose_mx, f_param_terms)
        
        s_dot = 1
        x_dot = fes(ca.vertcat(s,0,0))
        
        s_grid = np.linspace(self.s_min(), self.s_max(), 10000)
        ode    = {'x':ca.vertcat(s,x), 'ode':ca.vertcat(s_dot,x_dot)}
        config = {'t0':self.s_min(), 'tf':self.s_max(), 'grid':s_grid, 'output_t0':True, 'max_step_size':(s_grid[1] - s_grid[0])}
        xint  = ca.integrator('x','idas',ode,config)
        
        x_grid = np.array(xint(x0 = [self.s_min(),0,0,0])['xf'])
        
        xi = ca.interpolant('x','linear',[x_grid[0,:]],x_grid[1,:])
        xj = ca.interpolant('x','linear',[x_grid[0,:]],x_grid[2,:])
        xk = ca.interpolant('x','linear',[x_grid[0,:]],x_grid[3,:])
        
        xc = ca.vertcat(xi(s),xj(s),xk(s))
        self.sym_rep['xc'] = xc
    
    @abstractmethod
    def _post_centerline():
        '''
        intended for a centerline surface to turn 'xc' into 'xp', 
        ie. generate a sybolic expression for the shape of the surface.
        '''
        return
        
    def s_min(self, y = None):
        return self.config.s[0]
        
    def s_max(self, y = None):
        return self.config.s[-1]
        
    def generate_texture(self, S = None, Y = None, thickness = None, road_offset = 0, as_opengl = True, s_ext = 0, y_ext = 0):
        if S is None or Y is None:
            S,Y = self.parametric_grid(n_s = 1000, n_y = 20, s_ext = s_ext)
        return BaseSurface.generate_texture(self, S = S, Y = Y, thickness = thickness, road_offset = road_offset, as_opengl = as_opengl, y_ext = y_ext)   

class BaseLinearCenterlineSurface(BaseCenterlineSurface):
    
    '''
    class of surfaces that reference a centerline, and are cross-sectionally a line, ie. the Frenet surface and ribbon surfaces.
    '''
        
    def _post_centerline(self):
        ''' 
        obtain a function from parametric position to global position 
        for a linear centerline surface
        '''
        xc = self.sym_rep['xc']
        ey = self.sym_rep['ey']
        
        pose = self.sym_rep['pose']
        pose_mx = self.sym_rep['pose_mx']
        param_terms = self.sym_rep['param_terms']
        f_param_terms = self.sym_rep['f_param_terms']
        
        y = self.sym_rep['y']
        f_dy = ca.Function('ey', [pose, param_terms], [ey*y])  # general function for lateral offset
        dy = f_dy.call([pose_mx, f_param_terms(pose_mx)])[0]   # lateral offset for specific surface parameterization
        
        # add lateral offset to centerline and turn into a function
        xp = xc + dy
        f_xp = ca.Function('xp',[pose_mx],[xp])
        
        self.sym_rep['xp'] = xp
        self.sym_rep['f_xp'] = f_xp
        self.pose2xp = lambda pose: np.array(f_xp(pose))
        
