''' point mass models '''

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from vehicle_3d.pytypes import BaseBodyState, BodyLinearAcceleration
from vehicle_3d.models.dynamics_model import DynamicsModel, \
    DynamicsModelConfig, DynamicsModelVars
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.visualization.utils import get_unit_arrow, get_thrust_tf_ca, get_sphere
from vehicle_3d.visualization.objects import InstancedVertexObject, VertexObject, \
    UBOObject

@dataclass
class PointState(BaseBodyState):
    ''' point mass vehicle state '''
    u: BodyLinearAcceleration = field(default = None)

    def __post_init__(self):
        super().__post_init__()
        # link the two fields by default
        self.u = self.ab

@dataclass
class PointModelConfig(DynamicsModelConfig):
    ''' configuration of point mass model '''
    mu: float = field(default = 1.0)
    n: float = field(default = 0)
    a1_max: float = field(default = 10)
    a2_max: float = field(default = 100)

class TangentPointModel(DynamicsModel):
    ''' tangent contact point mass model '''
    config: PointModelConfig

    f_a_instances: ca.Function

    def __init__(self, config: PointModelConfig, surf: BaseSurface):
        super().__init__(config, surf)

    def _create_model(self):
        self.model_vars = DynamicsModelVars()
        s = self.surf.sym_rep.s
        y = self.surf.sym_rep.y
        v1 = ca.SX.sym('v1')
        v2 = ca.SX.sym('v2')
        w3 = 0
        a1 = ca.SX.sym('a1')
        a2 = ca.SX.sym('a2')

        s_dot, y_dot, ths_dot, w1, w2 = \
            self.surf.pose_eqns_2D(v1, v2, w3, self.config.n)

        s_dot = ca.substitute(s_dot, self.surf.sym_rep.ths, 0)
        y_dot = ca.substitute(y_dot, self.surf.sym_rep.ths, 0)
        ths_dot = ca.substitute(ths_dot, self.surf.sym_rep.ths, 0)
        w1 = ca.substitute(w1, self.surf.sym_rep.ths, 0)
        w2 = ca.substitute(w2, self.surf.sym_rep.ths, 0)

        w3 = -ths_dot

        R = ca.substitute(self.surf.sym_rep.R_ths, self.surf.sym_rep.ths, 0)
        e1 = R[:,0]
        e2 = R[:,1]
        e3 = R[:,2]
        ag1 = -self.config.g * e1[2]
        ag2 = -self.config.g * e2[2]
        ag3 = -self.config.g * e3[2]
        v1_dot = w3*v2 + a1 + ag1
        v2_dot =-w3*v1 + a2 + ag2

        a3 = v2*w1 - v1*w2 - ag3

        z = ca.vertcat(s, y, v1, v2)
        u = ca.vertcat(a1, a2)
        z_dot = ca.vertcat(s_dot, y_dot, v1_dot, v2_dot)

        # friction cone constraint
        g = ca.vertcat(
            a3,
            (a1**2 + a2**2) / (self.config.mu *a3)**2,
        )
        ubg = [np.inf, 1.]
        lbg = [0., -np.inf]

        self.model_vars.p = ca.vertcat(s, y, self.config.n)
        self.model_vars.R = R
        self.model_vars.vb = ca.vertcat(v1, v2, 0)
        self.model_vars.wb = ca.vertcat(w1, w2, w3)
        self.model_vars.ab = ca.vertcat(a1, a2, a3)

        self.model_vars.z = z
        self.model_vars.u = u
        self.model_vars.z_dot = z_dot
        self.model_vars.g = g
        self.model_vars.ubg = ubg
        self.model_vars.lbg = lbg

        self.f_R = self.surf.fill_in_param_terms(R, [z, u])

    def get_empty_state(self) -> PointState:
        return PointState()

    def state2u(self, state: PointState):
        return [state.ab.a1, state.ab.a2]

    def state2z(self, state: PointState):
        return [state.p.s, state.p.y, state.vb.v1, state.vb.v2]

    def u2state(self, state: PointState, u):
        state.u.a1 = u[0]
        state.u.a2 = u[1]

    def zu(self, s: float = 0):
        return [self.surf.s_max(), self.surf.y_max(s), np.inf, np.inf]

    def zl(self, s: float = 0):
        return [self.surf.s_min(), self.surf.y_min(s), -np.inf, -np.inf]

    def uu(self):
        return [self.config.a1_max, self.config.a2_max]

    def ul(self):
        return [-self.config.a1_max, -self.config.a2_max]

    def duu(self):
        return [np.inf, np.inf]

    def dul(self):
        return [-np.inf, -np.inf]

    def get_color(self):
        if self.config.build_planar_model:
            return [0.8, 0.8, 0.8, 1.0]
        return [0.6, 0.6, 0.6, 1.0]

    def get_label(self):
        if self.config.build_planar_model:
            return 'Planar Point Mass'
        return 'Point Mass'

    def generate_visual_assets(self, ubo: UBOObject):
        if self.visual_assets is not None:
            return self.visual_assets

        V, I = get_sphere(r = 0.3, color = self.get_color())
        point = VertexObject(ubo, V, I)

        # get normal force object and function to compute instance transforms
        V, I = get_unit_arrow(d=3, color = [1, 0, 0, 1])
        accelerations = InstancedVertexObject(ubo, V, I)

        a1_b = np.array([1,0,0]) * self.model_vars.ab[0]
        a2_b = np.array([0,1,0]) * self.model_vars.ab[1]
        a3_b = np.array([0,0,1]) * self.model_vars.ab[2]

        a1_instance_matrix = ca.SX_eye(4)
        a1_instance_matrix[:3,:3] = get_thrust_tf_ca(a1_b, norm=4)
        a2_instance_matrix = ca.SX_eye(4)
        a2_instance_matrix[:3,:3] = get_thrust_tf_ca(a2_b, norm=4)
        a3_instance_matrix = ca.SX_eye(4)
        a3_instance_matrix[:3,:3] = get_thrust_tf_ca(a3_b, norm=4)
        a_instances = ca.vertcat(
            a1_instance_matrix,
            a2_instance_matrix,
            a3_instance_matrix,
        )
        self.f_a_instances = self.surf.fill_in_param_terms(
            [a_instances],
            [self.model_vars.z, self.model_vars.u]
        )

        self.visual_assets = {
            'Point': point,
            'Accelerations': accelerations,
        }
        return self.visual_assets

    def update_visual_assets(self, state: BaseBodyState = None, dt: float = None):
        z, u = self.state2zu(state)
        for _, item in self.visual_assets.items():
            item.update_pose(state.x, state.q)
        self.visual_assets['Accelerations'].apply_instancing(
            np.array(self.f_a_instances(z,u)).reshape((-1,4,4)).astype(np.float32)
        )

    def get_instanced_visual_asset(self, ubo: UBOObject):
        V, I = get_sphere(r = 0.3, color = self.get_color())
        point = InstancedVertexObject(ubo, V, I)
        return point
