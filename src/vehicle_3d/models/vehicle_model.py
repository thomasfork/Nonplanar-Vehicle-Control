''' core vehicle model attributes (four wheeled vehicles) '''
from dataclasses import dataclass, field
from typing import Dict, List
from abc import abstractmethod

import casadi as ca
import numpy as np

from vehicle_3d.pytypes import VectorizablePythonMsg, BaseBodyState, BodyMoment, BodyForce,\
    NestedPythonMsg
from vehicle_3d.utils.ca_utils import ca_abs
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.dynamics_model import DynamicsModel, DynamicsModelConfig, \
    DynamicsModelVars, DAEDynamicsModel, DAEDynamicsModelVars
from vehicle_3d.models.tire_model import Tire, TireConfig, TireState

from vehicle_3d.utils.load_utils import get_assets_file
from vehicle_3d.visualization.shaders import vtype
from vehicle_3d.visualization.utils import get_unit_arrow
from vehicle_3d.visualization.ca_glm import scale, rotate
from vehicle_3d.visualization.objects import InstancedVertexObject, VertexObject, UBOObject
from vehicle_3d.visualization.gltf2 import load_car, TargetObjectSize, GLTFObject

@dataclass
class BaseVehicleActuation(VectorizablePythonMsg):
    ''' general purpose high-level vehicle actuation '''
    a: float = field(default =0.)
    ''' throttle command '''
    b: float = field(default =0.)
    ''' brake command '''
    y: float = field(default = 0.)
    ''' front steering angle command '''

    def to_vec(self):
        return np.array([self.a, self.b, self.y])

    def from_vec(self, vec: np.ndarray):
        self.a, self.b, self.y = vec

@dataclass
class BaseVehicleState(BaseBodyState):
    ''' fields common to all vehicles '''
    u: BaseVehicleActuation = field(default = None)
    Fg: BodyForce = field(default = None)
    Fd: BodyForce = field(default = None)
    Kd: BodyMoment = field(default = None)

    tfr: TireState = field(default = None)
    tfl: TireState = field(default = None)
    trr: TireState = field(default = None)
    trl: TireState = field(default = None)

@dataclass
class VehicleModelConfig(DynamicsModelConfig, NestedPythonMsg):
    ''' config for any vehicle model '''
    m: float = field(default = 1303)
    '''vehicle mass in kg'''
    I1: float = field(default = 1000)
    '''roll inertia  (w1)'''
    I2: float = field(default = 4000)
    '''pitch inertia (w2)'''
    I3: float = field(default = 4500)
    '''yaw inertia   (w3)'''
    lf: float = field(default = 1.521)
    '''distance from COM to front axle'''
    lr: float = field(default = 1.499)
    '''distance from COM to rear axle'''

    @property
    def L(self) -> float:
        ''' wheelbase length '''
        return self.lf + self.lr

    tf: float = field(default = 0.9)
    '''disance from COM to left/right front wheels'''
    tr: float = field(default = 0.9)
    '''disance from COM to left/right rear wheels'''
    h:  float = field(default = 0.592)
    '''distance from road surface to COM'''

    w  :float = field(default =    2.6)
    '''width of the car'''
    bf :float = field(default =    2.9)
    '''distance from COM to front bumper'''
    br :float = field(default =    2.45)
    '''distance from COM to rear bumper'''
    ht :float = field(default =    1.25)
    '''total height of the car'''

    c1: float = field(default = 0.)
    ''' quadratic longitudinal drag '''
    c2: float = field(default = 0.)
    ''' quadtratic lateral drag '''
    c3: float = field(default = 0.)
    ''' quadratic downforce from longitudinal vel.'''
    b1: float = field(default = 0.)
    ''' linear longitudinal drag '''
    b2: float = field(default = 0.)
    ''' linear lateral drag '''

    ua_max: float = field(default = 10)
    ua_min: float = field(default = 0)
    dua_max: float = field(default = 50)
    dua_min: float = field(default = -50)
    ub_max: float = field(default = 10)
    ub_min: float = field(default = 0)
    dub_max: float = field(default = 50)
    dub_min: float = field(default = -50)
    uy_max: float = field(default = 0.6)
    uy_min: float = field(default =-0.6)
    duy_max: float = field(default = 2.5)
    duy_min: float = field(default =-2.5)

    N_max: float = field(default = 40000)
    tire_config: TireConfig = field(default = None)

    def force_normalization_const(self):
        '''
        return a constant to normalize forces by
        intended for internal model use
        so that optimization variables are not poorly scaled for heavy vehicles
        '''
        return self.m * self.g

    def get_polytope(self):
        ''' returns an external bounding polytope for the vehicle'''
        A = np.concatenate([np.eye(3), -np.eye(3)])
        b = np.array([self.bf, self.w/2, self.h, self.br, self.w/2, self.h])
        return A, b

    def get_drawable_polytope(self, ubo: UBOObject):
        ''' generate data for plotting vehicle outline in opengl '''
        e = np.array([[1,1,1,],
                      [1,1,-1],
                      [1,-1,1],
                      [-1,1,1],
                      [1,-1,-1],
                      [-1,1,-1],
                      [-1,-1,1],
                      [-1,-1,-1]]).T

        x =  e * np.array([[(self.bf + self.br)/2, self.w/2, self.h]]).T + \
            np.array([[(self.bf - self.br)/2, 0, 0]]).T
        x = x.T

        Vertices = np.zeros(x.shape[0], dtype=vtype)
        Vertices['a_position'] = x
        Vertices['a_normal'] = np.array([0,0,1])
        Vertices['a_color'] = [0.8, 0.8, 0.8, 1]
        I = np.array([0,1,0,2,0,3,1,4,1,5,2,4,2,6,3,5,3,6,4,7,5,7,6,7], dtype = np.uint32)
        return VertexObject(ubo, Vertices, I, simple=True, lines = True)

    @classmethod
    def barc_defaults(cls):
        ''' default vehicle config for barc '''
        config = cls()
        config.m = 2.2187
        config.I1 = 0.01
        config.I2 = 0.035
        config.I3 = 0.027
        config.lf = 0.13
        config.lr = 0.13
        config.tf = 0.1
        config.tr = 0.1
        config.h = 0.05
        config.w = 0.2
        config.bf = 0.15
        config.br = 0.15
        config.ht = 0.15
        config.tire_config = TireConfig()
        config.tire_config.m = 0.050
        config.tire_config.I = 2e-5
        config.tire_config.r = 0.024

        return config

@dataclass
class VehicleModelVars(DynamicsModelVars):
    ''' expected variables from any vehicle model '''
    ua: ca.SX = field(default = None)
    uy: ca.SX = field(default = None)
    ub: ca.SX = field(default = None)

    s: ca.SX = field(default = None)
    s_dot: ca.SX = field(default = None)
    y: ca.SX = field(default = None)
    y_dot: ca.SX = field(default = None)
    n: ca.SX = field(default = None)
    n_dot: ca.SX = field(default = None)

    v1: ca.SX = field(default = None)
    v2: ca.SX = field(default = None)
    v3: ca.SX = field(default = None)

    w1: ca.SX = field(default = None)
    w2: ca.SX = field(default = None)
    w3: ca.SX = field(default = None)

    v1_dot: ca.SX = field(default = None)
    v2_dot: ca.SX = field(default = None)
    v3_dot: ca.SX = field(default = None)

    w1_dot: ca.SX = field(default = None)
    w2_dot: ca.SX = field(default = None)
    w3_dot: ca.SX = field(default = None)

    N: ca.SX = field(default = None)
    ''' tire normal forces, may be normalized '''
    N_reg: ca.SX = field(default = None)
    ''' tire normal forces, not normalized '''

    F: ca.SX = field(default = None)
    K: ca.SX = field(default = None)
    Fg: ca.SX = field(default = None)
    Fd: ca.SX = field(default = None)
    Kd: ca.SX = field(default = None)
    Ft: ca.SX = field(default = None)
    Kt: ca.SX = field(default = None)


class VehicleModel(DynamicsModel):
    ''' base vehicle model class '''
    config: VehicleModelConfig
    model_vars: VehicleModelVars

    f_N: ca.Function
    f_tfr: ca.Function
    f_tfl: ca.Function
    f_trr: ca.Function
    f_trl: ca.Function
    f_F: ca.Function
    f_K: ca.Function
    f_Fg: ca.Function
    f_Fd: ca.Function
    f_Kd: ca.Function
    f_Ft: ca.Function
    f_Kt: ca.Function

    tires: List[Tire] = None
    # functions to unpack tire state
    f_tfr: ca.Function
    f_tfl: ca.Function
    f_trr: ca.Function
    f_trl: ca.Function

    # helper functions for 3d rendering
    # populated by self.generate_visual_assets
    f_N_instances: ca.Function
    f_Ft_instances: ca.Function

    def __init__(self, config: VehicleModelConfig, surf: BaseSurface):
        super().__init__(config, surf)

    def step(self, state: BaseVehicleState):
        self._step_tire_rotation(state, dt = self.config.dt/2)
        super().step(state)
        self._step_tire_rotation(state, dt = self.config.dt/2)

    def _step_tire_rotation(self, state: BaseVehicleState, dt = None):
        '''
        internal function to increment tire rotation variables
        '''
        if dt is None:
            dt = self.config.dt
        state.tfr.th += state.tfr.w * dt
        state.tfl.th += state.tfl.w * dt
        state.trr.th += state.trr.w * dt
        state.trl.th += state.trl.w * dt

    def _create_model(self):
        self._clear_model_vars()
        self._add_input_vars()
        self._add_pose_vars()
        self._add_vel_vars()
        self._add_forces()
        self._state_derivative()
        self._calc_outputs()

    @abstractmethod
    def _clear_model_vars(self):
        ''' reset self.model_vars'''

    def _add_input_vars(self):
        # override for models that don't use all of these
        self.model_vars.uy = ca.SX.sym('uy')
        self.model_vars.ua = ca.SX.sym('ua')
        self.model_vars.ub = ca.SX.sym('ub')
        self.model_vars.u = ca.vertcat(
            self.model_vars.ua,
            self.model_vars.ub,
            self.model_vars.uy)

    @abstractmethod
    def _add_pose_vars(self):
        ''' create pose and orientation variables '''

    @abstractmethod
    def _add_vel_vars(self):
        '''
        create linear and angular velocity variables
        as well as their parametric equilvalents (s_dot, etc..)
        '''

    def _add_forces(self):
        self._grav_forces()
        self._drag_forces()
        self._tire_forces()
        self.model_vars.F = self.model_vars.Fg + self.model_vars.Fd + self.model_vars.Ft
        self.model_vars.K = self.model_vars.Kd + self.model_vars.Kt
        self.model_vars.ab = (self.model_vars.F - self.model_vars.Fg) / self.config.m

    def _grav_forces(self):
        if self.surf.config.flat or self.config.build_planar_model:
            Fg1 = 0
            Fg2 = 0
            Fg3 = -self.config.m * self.config.g
        else:
            R = self.model_vars.R
            e1 = R[:,0]
            e2 = R[:,1]
            e3 = R[:,2]
            Fg1 = -self.config.m * self.config.g * e1[2]
            Fg2 = -self.config.m * self.config.g * e2[2]
            Fg3 = -self.config.m * self.config.g * e3[2]

        self.model_vars.Fg = ca.vertcat(Fg1, Fg2, Fg3)

    def _drag_forces(self):
        v1 = self.model_vars.vb[0]
        v2 = self.model_vars.vb[1]

        Fd1 = -self.config.c1 * v1 * ca_abs(v1) \
            - self.config.b1 * v1
        Fd2 = -self.config.c2 * v2 * ca_abs(v2)
        Fd3 = -self.config.c3 * v1 * v1
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0

        self.model_vars.Fd = ca.vertcat(Fd1, Fd2, Fd3)
        self.model_vars.Kd = ca.vertcat(Kd1, Kd2, Kd3)

    @abstractmethod
    def _tire_forces(self):
        ''' add tire forces '''

    @abstractmethod
    def _state_derivative(self):
        ''' set up state and state derivative '''

    @abstractmethod
    def _calc_outputs(self):
        ''' compute model outputs and limits thereof '''

    def _setup_helper_functions(self):
        super()._setup_helper_functions()
        args = self.model_vars.get_all_indep_vars()
        self.f_N = self.surf.fill_in_param_terms(
            self.model_vars.N_reg,
            args
        )
        self.f_F = self.surf.fill_in_param_terms(
            self.model_vars.F,
            args
        )
        self.f_K = self.surf.fill_in_param_terms(
            self.model_vars.K,
            args
        )
        self.f_Fg = self.surf.fill_in_param_terms(
            self.model_vars.Fg,
            args
        )
        self.f_Fd = self.surf.fill_in_param_terms(
            self.model_vars.Fd,
            args
        )
        self.f_Kd = self.surf.fill_in_param_terms(
            self.model_vars.Kd,
            args
        )
        self.f_Ft = self.surf.fill_in_param_terms(
            self.model_vars.Ft,
            args
        )
        self.f_Kt = self.surf.fill_in_param_terms(
            self.model_vars.Kt,
            args
        )

        self.f_tfr = self.surf.fill_in_param_terms(
            self.tires[0].get_tire_state_vec(),
            args
        )
        self.f_tfl = self.surf.fill_in_param_terms(
            self.tires[1].get_tire_state_vec(),
            args
        )
        self.f_trr = self.surf.fill_in_param_terms(
            self.tires[2].get_tire_state_vec(),
            args
        )
        self.f_trl = self.surf.fill_in_param_terms(
            self.tires[3].get_tire_state_vec(),
            args
        )

    def state2u(self, state: BaseVehicleState):
        return state.u.to_vec().tolist()

    def u2state(self, state: BaseVehicleState, u):
        u = self._coerce_input_limits(u)
        state.u.from_vec(u)

    @abstractmethod
    def zu2state(self, state: BaseVehicleState, z, u):
        # abstract as this is missing relative orientation
        super().zu2state(state, z, u)
        state.tfr.from_vec(self.f_tfr(z,u))
        state.tfl.from_vec(self.f_tfl(z,u))
        state.trr.from_vec(self.f_trr(z,u))
        state.trl.from_vec(self.f_trl(z,u))
        state.Fg.from_vec(self.f_Fg(z, u))
        state.Fd.from_vec(self.f_Fd(z, u))
        state.Kd.from_vec(self.f_Kd(z, u))

    def uu(self):
        return [
            self.config.ua_max,
            self.config.ub_max,
            self.config.uy_max
        ]

    def ul(self):
        return [
            self.config.ua_min,
            self.config.ub_min,
            self.config.uy_min
        ]

    def duu(self):
        return [
            self.config.dua_max,
            self.config.dub_max,
            self.config.duy_max
        ]

    def dul(self):
        return [
            self.config.dua_min,
            self.config.dub_min,
            self.config.duy_min
        ]

    def generate_visual_assets(self, ubo: UBOObject) -> Dict[str, VertexObject]:
        if self.visual_assets is not None:
            return self.visual_assets
        inputs = self.model_vars.get_all_indep_vars()

        car = AnimatedCar(ubo, self.config, color=self.get_color())

        # get normal force object and function to compute instance transforms
        V, I = get_unit_arrow(d=3, color = [1, 0, 0, 1],
                              ri = self.config.L / 60,
                              ro = self.config.L / 30)
        normal_forces = InstancedVertexObject(ubo, V, I)
        normal_force_instances = ca.vertcat(*(
            tire.normal_force_instance_matrix for tire in self.tires
        ))
        self.f_N_instances = self.surf.fill_in_param_terms(
            normal_force_instances,
            inputs
        )

        # get tire force object and function to compute instance transforms
        tire_forces = InstancedVertexObject(ubo, V, I)
        tire_foce_instances = ca.vertcat(*(
            tire.tire_force_instance_matrix for tire in self.tires
        ))
        self.f_Ft_instances = self.surf.fill_in_param_terms(
            tire_foce_instances,
            inputs
        )

        outline = self.config.get_drawable_polytope(ubo)

        self.visual_assets = {
            'Car': car,
            'Collision Boundaries': outline,
            'Normal Forces': normal_forces,
            'Tire Forces': tire_forces,
        }
        return self.visual_assets

    def update_visual_assets(self, state: BaseVehicleState, dt: float = None):
        self._step_tire_rotation(state, dt = dt)
        if isinstance(self, DAEDynamicsModel):
            inputs = self.state2zua(state)
        else:
            inputs = self.state2zu(state)
        self.visual_assets['Car'].update(state)
        for _, item in self.visual_assets.items():
            item.update_pose(state.x, state.q)
        self.visual_assets['Normal Forces'].apply_instancing(
            np.array(self.f_N_instances(*inputs)).reshape((-1,4,4)).astype(np.float32)
        )
        self.visual_assets['Tire Forces'].apply_instancing(
            np.array(self.f_Ft_instances(*inputs)).reshape((-1,4,4)).astype(np.float32)
        )

    def get_instanced_visual_asset(self, ubo: UBOObject):
        alpha = self.config.L / (1.025 + 1.28)
        d =2.00* alpha
        size = TargetObjectSize(
            fixed_aspect = True,
            max_dims=[d,d,d],
            min_dims=[-d,-d,-self.config.h])
        return load_car(ubo, color = self.get_color(), instanced = True, size = size)


class DAEVehicleModelVars(VehicleModelVars, DAEDynamicsModelVars):
    ''' model variables for DAE vehicles '''


class DAEVehicleModel(VehicleModel, DAEDynamicsModel):
    ''' base DAE vehicle model class '''

    @abstractmethod
    def zua2state(self, state: BaseVehicleState, z, u, a):
        # abstract as this is missing relative orientation
        super().zua2state(state, z, u, a)
        state.tfr.from_vec(self.f_tfr(z,u,a))
        state.tfl.from_vec(self.f_tfl(z,u,a))
        state.trr.from_vec(self.f_trr(z,u,a))
        state.trl.from_vec(self.f_trl(z,u,a))
        state.Fg.from_vec(self.f_Fg(z, u, a))
        state.Fd.from_vec(self.f_Fd(z, u, a))
        state.Kd.from_vec(self.f_Kd(z, u, a))


class AnimatedCar:
    ''' class to track and update objects to render for animated 3d vehicle '''
    objects: Dict[str, VertexObject]
    def __init__(self, ubo: UBOObject, config: VehicleModelConfig, color: list = None):

        lf = config.lf
        lr = config.lr

        alpha = config.L / (1.025 + 1.28)
        rear_tire_r = 0.32 * alpha
        front_tire_r = 0.29 * alpha

        tf = config.tf
        tr = config.tr

        Y = ca.SX.sym('y', 4)
        TH = ca.SX.sym('th', 4)

        y = Y[0]
        th = TH[0] * 180 / np.pi
        wheel_scale = scale(ca.DM_eye(4),front_tire_r, front_tire_r, front_tire_r)
        turn = rotate(ca.DM_eye(4), y*180 / ca.pi, 0, 0, 1)
        rot = rotate(ca.DM_eye(4), -th, 0, 0, 1)
        orient = rotate(ca.DM_eye(4), 90, 1, 0, 0)

        M_fr = turn @ orient @ rot @ wheel_scale
        M_fr[0,3] = lf
        M_fr[1,3] =-tf
        M_fr[2,3] = front_tire_r - config.h
        Mc_fr = turn @ orient @ wheel_scale
        Mc_fr[0,3] = lf
        Mc_fr[1,3] =-tf
        Mc_fr[2,3] = front_tire_r - config.h

        y = Y[2]
        th = TH[2] * 180 / np.pi
        rot = rotate(ca.DM_eye(4), -th, 0, 0, 1)
        turn = rotate(ca.DM_eye(4), y*180 / ca.pi, 0, 0, 1)
        wheel_scale = scale(ca.DM_eye(4),rear_tire_r, rear_tire_r, rear_tire_r)
        M_rr = turn @ orient @ rot @ wheel_scale
        M_rr[0,3] = -lr
        M_rr[1,3] =-tr
        M_rr[2,3] = rear_tire_r - config.h
        Mc_rr = turn @ orient @ wheel_scale
        Mc_rr[0,3] = -lr
        Mc_rr[1,3] =-tr
        Mc_rr[2,3] = rear_tire_r - config.h

        y = Y[1]
        th = TH[1] * 180 / np.pi
        wheel_scale = scale(ca.DM_eye(4),front_tire_r, front_tire_r, -front_tire_r)
        turn = rotate(ca.DM_eye(4), y*180 / ca.pi, 0, 0, 1)
        rot = rotate(ca.DM_eye(4), -th, 0, 0, 1)
        orient = rotate(ca.DM_eye(4), 90, 1, 0, 0)

        M_fl = turn @ orient @ rot @ wheel_scale
        M_fl[0,3] = lf
        M_fl[1,3] = tf
        M_fl[2,3] = front_tire_r - config.h
        Mc_fl = turn @ orient @ wheel_scale
        Mc_fl[0,3] = lf
        Mc_fl[1,3] = tf
        Mc_fl[2,3] = front_tire_r - config.h

        y = Y[3]
        th = TH[3] * 180 / np.pi
        wheel_scale = scale(ca.DM_eye(4),rear_tire_r, rear_tire_r, -rear_tire_r)
        turn = rotate(ca.DM_eye(4), y*180 / ca.pi, 0, 0, 1)
        rot = rotate(ca.DM_eye(4), -th, 0, 0, 1)
        M_rl = turn @ orient @ rot @ wheel_scale
        M_rl[0,3] = -lr
        M_rl[1,3] = tr
        M_rl[2,3] = rear_tire_r - config.h
        Mc_rl = turn @ orient @ wheel_scale
        Mc_rl[0,3] = -lr
        Mc_rl[1,3] = tr
        Mc_rl[2,3] = rear_tire_r - config.h

        inputs = ca.vertcat(
            Y,TH
        )

        M = ca.vertcat(M_fr, M_rr, M_fl, M_rl)
        f_M = ca.Function('M',[inputs],[M])
        Mc = ca.vertcat(Mc_fr, Mc_rr, Mc_fl, Mc_rl)
        f_Mc = ca.Function('Mc',[inputs],[Mc])

        wheel_size = TargetObjectSize(max_dims=[1,1,1],min_dims=[-1,-1,-1],
                                      squish_z=False, fixed_aspect = True, )

        wheels = GLTFObject(get_assets_file('lam_wheel.glb'), ubo, wheel_size,
                            cull_faces=False,
                            instanced=True,
                    ignored_nodes=['Object_6', 'Object_8','Object_9','Object_10','Object_20'])

        calipers = GLTFObject(get_assets_file('lam_wheel.glb'), ubo, wheel_size,
                            cull_faces=False,
                            instanced=True,
                    ignored_nodes=['Object_7', 'Object_11','Object_13','Object_15','Object_22',
                                    'Object_23','Object_4','Object_17', 'Object_18'])
        d =2.00* alpha
        delta = lf - lr + 0.15*alpha
        lambo_size = TargetObjectSize(
            fixed_aspect = True,
            max_dims=[d+delta,d,d],
            min_dims=[-d+delta,-d,-config.h])

        if color is None:
            lambo = GLTFObject(get_assets_file('lambo.glb'), ubo, lambo_size, cull_faces=False,
                            ignored_nodes=[f'Object_{k}' for k in
                                            [8,11,17,18,23,24,25,26,27,34,35,36,37,38,39]],
                                )
        else:
            lambo = GLTFObject(get_assets_file('lambo.glb'), ubo, lambo_size, cull_faces=False,
                            ignored_nodes=[f'Object_{k}' for k in
                                            [8,11,17,18,23,24,25,26,27,34,35,36,37,38,39]],
                                matl_color_swaps={'Material.012':color}
                                )

        self.f_M = f_M
        self.f_Mc = f_Mc

        self.objects = {
            'Lambo': lambo,
            'Calipers': calipers,
            'Wheels': wheels
        }
        self.wheels = wheels
        self.calipers = calipers
        self.lambo = lambo

    def get_objects(self) -> Dict[str, VertexObject]:
        ''' get objects to add to window and corresponding labels '''
        return self.objects

    def update(self, state: BaseVehicleState):
        ''' update the car '''
        inputs = [
            state.tfr.y,
            state.tfl.y,
            state.trr.y,
            state.trl.y,
            state.tfr.th,
            state.tfl.th,
            state.trr.th,
            state.trl.th,
        ]
        M = np.array(self.f_M(inputs)).reshape((-1,4,4)).astype(np.float32)
        Mc = np.array(self.f_Mc(inputs)).reshape((-1,4,4)).astype(np.float32)

        self.wheels.apply_instancing(M)
        self.calipers.apply_instancing(Mc)

    def draw(self):
        ''' draw the car'''
        for _, item in self.objects.items():
            item.draw()

    def update_pose(self, x = None, q = None, mat = None):
        ''' update pose of the car'''
        for _, item in self.objects.items():
            item.update_pose(x,q,mat)
