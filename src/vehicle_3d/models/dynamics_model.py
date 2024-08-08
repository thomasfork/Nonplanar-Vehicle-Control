'''
dynamics model template
'''
from abc import abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union

import casadi as ca
import numpy as np

from vehicle_3d.pytypes import PythonMsg, BaseBodyState
from vehicle_3d.utils.integrators import idas_integrator, \
    idas_dae_integrator
from vehicle_3d.surfaces.base_surface import BaseSurface
from vehicle_3d.models.model import StatefulModelVars, StatefulModel, \
    DAEModelVars, StatefulDAEModel

from vehicle_3d.visualization.objects import VertexObject, \
    InstancedVertexObject, UBOObject

@dataclass
class DynamicsModelConfig(PythonMsg):
    ''' expected configuration variables with any dynamics model '''
    dt: float = field(default = 0.01)
    ''' simulation interval'''
    g:  float = field(default = 9.81)
    ''' gravitational acceleration magnitude '''

    build_planar_model: bool = field(default = False)
    # will cause the model to treat the surface as flat

    print_model_warnings: bool = field(default = True)

@dataclass
class DynamicsModelVars(StatefulModelVars):
    ''' expected variables from any dynamics model '''
    p: ca.SX = field(default = None)
    R: ca.SX = field(default = None)
    vb: ca.SX = field(default = None)
    wb: ca.SX = field(default = None)
    ab: ca.SX = field(default = None)

@dataclass
class DAEDynamicsModelVars(DynamicsModelVars, DAEModelVars):
    ''' expected variables for DAE dynamics model '''


class DynamicsModel(StatefulModel):
    ''' base dynamics model '''
    config: DynamicsModelConfig
    model_vars: DynamicsModelVars
    surf: BaseSurface

    f_zdot: ca.Function
    f_zdot_full: ca.Function
    f_znew: ca.Function
    f_g   : ca.Function
    f_g_full: ca.Function

    f_p: ca.Function
    f_R: ca.Function
    f_vb: ca.Function
    f_wb: ca.Function
    f_ab: ca.Function

    # objects and helper functions for 3d rendering
    # populated by self.generate_visual_assets
    # updated by self.update_visual_assets
    visual_assets: Dict[str, Union[VertexObject, InstancedVertexObject]] = None

    def __init__(self, config: DynamicsModelConfig, surf: BaseSurface):
        self.config = config
        self.surf = surf
        self._pre_setup_checks()
        self._create_model()
        self._model_setup_checks()
        self._setup_integrator()
        self._setup_helper_functions()

    def step(self, state: BaseBodyState) -> None:
        ''' forward pass of dynamics model '''
        z, u = self.state2zu(state)
        u = self._coerce_input_limits(u)
        zn = self.f_znew(z, u)
        self._check_model_output_limits((zn,u))
        self.zu2state(state, zn, u)
        state.t += self.config.dt

    def _coerce_input_limits(self, u: List[float]) -> List[float]:
        ''' coerce an input vector to allowed range'''
        return np.clip(u, self.ul(), self.uu())

    def _check_model_output_limits(self, inputs):
        if not self.config.print_model_warnings:
            return
        if self.model_vars.ubg:
            g = self.f_g(*inputs)
            if any(g > self.model_vars.ubg) or any(g < self.model_vars.lbg):
                print('WARNING - Model outputs exceeded during simulation')

    def _pre_setup_checks(self):
        ''' any checks that need to be run before trying to create a model'''

    @abstractmethod
    def _create_model(self):
        ''' create the model '''

    def _model_setup_checks(self):
        self.model_vars.state_dim = self.model_vars.z.shape[0]
        self.model_vars.input_dim = self.model_vars.u.shape[0]
        if isinstance(self, DAEDynamicsModel):
            self.model_vars.alg_dim = self.model_vars.a.shape[0]
            self.model_vars.dae_dim = self.model_vars.h.shape[0]
        for label, attr in asdict(self.model_vars).items():
            if attr is None:
                raise RuntimeError(f'Incomplete model setup, attribute {label} is NoneType')

    def _setup_integrator(self):
        ''' set up integrator '''
        inputs = self.model_vars.get_all_indep_vars()

        # dispatch disabled so that casadi function size is visible
        self.f_zdot = self.surf.fill_in_param_terms(
            self.model_vars.z_dot,
            inputs,
        )
        self.f_zdot_full = ca.Function('z_dot',
            inputs + [self.surf.sym_rep.param_terms],
            [self.model_vars.z_dot],
        )
        self.f_g = self.surf.fill_in_param_terms(
            self.model_vars.g,
            inputs
        )
        self.f_g_full = ca.Function('g',
            inputs + [self.surf.sym_rep.param_terms],
            [self.model_vars.g],
        )

        if isinstance(self, DAEDynamicsModel):
            self.f_h = self.surf.fill_in_param_terms(
                self.model_vars.h,
                inputs
            )
            self.f_h_full = ca.Function('h',
                inputs + [self.surf.sym_rep.param_terms],
                [self.model_vars.h],
            )
            self.f_znew, self.f_zanew = idas_dae_integrator(
                *self.model_vars.get_all_indep_vars(),
                self.f_zdot(*inputs),
                self.f_h(*inputs),
                self.config.dt
            )
        else:
            self.f_znew = idas_integrator(
                *self.model_vars.get_all_indep_vars(),
                self.f_zdot(*inputs),
                self.config.dt
            )

    def _setup_helper_functions(self):
        args = self.model_vars.get_all_indep_vars()
        self.f_p = self.surf.fill_in_param_terms(
            self.model_vars.p,
            args
        )
        self.f_R = self.surf.fill_in_param_terms(
            self.model_vars.R,
            args
        )
        self.f_vb = self.surf.fill_in_param_terms(
            self.model_vars.vb,
            args
        )
        self.f_wb = self.surf.fill_in_param_terms(
            self.model_vars.wb,
            args
        )
        self.f_ab = self.surf.fill_in_param_terms(
            self.model_vars.ab,
            args
        )

    @abstractmethod
    def get_empty_state(self) -> BaseBodyState:
        pass

    def zu2state(self, state: BaseBodyState, z, u):
        self.u2state(state, u)
        state.p.from_vec(self.f_p(z, u))
        self.surf.wrap_s(state)
        state.q.from_mat(self.f_R(z, u))
        state.vb.from_vec(self.f_vb(z, u))
        state.wb.from_vec(self.f_wb(z, u))
        state.ab.from_vec(self.f_ab(z, u))
        self.surf.p2gx(state)

    def add_model_stage_constraints(self, inputs, g, lbg, ubg, param_terms = None):
        ''' helper to add constraints for building optimization problems '''
        if param_terms is not None:
            g_eval = self.f_g_full(*inputs, param_terms)
        else:
            g_eval = self.f_g(*inputs)
        if g_eval is not None:
            g += [g_eval]
            lbg += self.model_vars.lbg
            ubg += self.model_vars.ubg

    @abstractmethod
    def get_color(self) -> List[float]:
        ''' return default color for vehicles using this model '''

    @abstractmethod
    def get_label(self) -> str:
        ''' return default label for vehicles using this model '''

    @abstractmethod
    def generate_visual_assets(self, ubo: UBOObject) -> Dict[str, VertexObject]:
        ''' generate assets for 3d rendering of the vehicle model '''

    @abstractmethod
    def update_visual_assets(self, state: BaseBodyState, dt: float = None):
        ''' update 3d assets for rendering the vehicle model at given state and input '''

    @abstractmethod
    def get_instanced_visual_asset(self, ubo: UBOObject) -> InstancedVertexObject:
        ''' get an instanced object for rendering many copies of the object '''


class DAEDynamicsModel(DynamicsModel, StatefulDAEModel):
    ''' base class for dynamics model with DAE setup '''
    model_vars: DAEDynamicsModelVars

    f_zanew: ca.Function
    f_h: ca.Function
    f_h_full: ca.Function

    def step(self, state: BaseBodyState) -> None:
        ''' forward pass of dynamics model '''
        z, u, a  = self.state2zua(state)
        u = self._coerce_input_limits(u)
        zn, af = self.f_zanew(z, u, a)
        self._check_model_output_limits((zn,u,a))
        self.zua2state(state, zn, u, af)
        state.t += self.config.dt

    def zu2state(self, state: BaseBodyState, z, u):
        raise NotImplementedError('DAE Requires algebraic state vector as well')

    def zua2state(self, state: BaseBodyState, z, u, a):
        self.u2state(state, u)
        state.p.from_vec(self.f_p(z, u, a))
        self.surf.wrap_s(state)
        state.q.from_mat(self.f_R(z, u, a))
        state.vb.from_vec(self.f_vb(z, u, a))
        state.wb.from_vec(self.f_wb(z, u, a))
        state.ab.from_vec(self.f_ab(z, u, a))
        self.surf.p2gx(state)
