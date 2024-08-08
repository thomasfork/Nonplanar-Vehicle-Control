'''
Standard types ie. position, state and input
'''
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import get_type_hints
import copy
import inspect

import dill
import numpy as np
import casadi as ca
import scipy.spatial.transform as transform

@dill.register(ca.SX)
def _dill_sx_compat(pickler, obj: ca.SX):
    ''' teach dill how to serialize casadi SX '''
    pickler.save_reduce(ca.SX.deserialize, (obj.serialize(),), obj=obj)

def _mx_serialize(x: ca.MX):
    _ca_serializer = ca.StringSerializer()
    _ca_serializer.pack(x)
    return _ca_serializer.encode()

def _mx_deserialize(s: str):
    _ca_deserializer = ca.StringDeserializer(s)
    return _ca_deserializer.unpack()

@dill.register(ca.MX)
def _dill_mx_compat(pickler, obj: ca.MX):
    ''' teach dill how to serialize casadi MX '''
    pickler.save_reduce(_mx_deserialize, (_mx_serialize(obj),), obj=obj)

@dataclass
class PythonMsg:
    '''
    base class for creating types and messages in python
    '''
    def __setattr__(self,key,value):
        '''
        Overloads default atribute-setting functionality
          to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally
          adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self,key):
            raise TypeError (f'Not allowed to add new field "{key}" to class {self}')
        else:
            object.__setattr__(self,key,value)

    def copy(self):
        ''' creates a copy of this class instance'''
        return copy.deepcopy(self)

    def pprint(self, indent = 0):
        ''' a more pretty way to print data '''
        indent = max(indent, 0)

        if indent == 0:
            print(' ' * indent + type(self).__name__)
        else:
            print(f'({type(self).__name__})')

        indent += 2
        for key in vars(self):
            attr = getattr(self, key)
            if isinstance(attr, PythonMsg):
                print(' ' * indent + f'{key} : ', end='')
                attr.pprint(indent = indent)
            else:
                print(' ' * indent + f'{key} : {attr}')

@dataclass
class VectorizablePythonMsg(PythonMsg, ABC):
    ''' structure that can be converted to/from a vector '''
    @abstractmethod
    def to_vec(self) -> np.ndarray:
        ''' convert structure to vector '''

    @abstractmethod
    def from_vec(self, vec: np.ndarray) -> None:
        ''' update structure from vector '''

@dataclass
class NestedPythonMsg(PythonMsg):
    '''
    structure that is nested
    adds helper to create classes, set default values to None to use
    '''

    def __post_init__(self):
        for ancestor_class in type(self).mro():
            for name, type_hint in get_type_hints(ancestor_class).items():
                if not inspect.isclass(type_hint):
                    continue
                elif not issubclass(type_hint, PythonMsg):
                    continue
                elif getattr(self, name) is None:
                    setattr(self, name, type_hint())

@dataclass
class Position(VectorizablePythonMsg):
    ''' 3D position in the global frame'''
    xi: float = field(default = 0.)
    xj: float = field(default = 0.)
    xk: float = field(default = 0.)

    def xdot(self, q: 'OrientationQuaternion', v: 'BodyLinearVelocity') -> 'Position':
        ''' position derivative from orientation and body frame velocity'''
        # pylint: disable=line-too-long
        xdot = Position()
        xdot.xi = (1 - 2*q.qj**2 - 2*q.qk**2)*v.v1 + 2*(q.qi*q.qj - q.qk*q.qr)*v.v2 + 2*(q.qi*q.qk + q.qj*q.qr)*v.v3
        xdot.xj = (1 - 2*q.qk**2 - 2*q.qi**2)*v.v2 + 2*(q.qj*q.qk - q.qi*q.qr)*v.v3 + 2*(q.qj*q.qi + q.qk*q.qr)*v.v1
        xdot.xk = (1 - 2*q.qi**2 - 2*q.qj**2)*v.v3 + 2*(q.qk*q.qi - q.qj*q.qr)*v.v1 + 2*(q.qk*q.qj + q.qi*q.qr)*v.v2
        return xdot

    def to_vec(self):
        return np.array([self.xi, self.xj, self.xk])

    def from_vec(self, vec):
        self.xi, self.xj, self.xk = vec

@dataclass
class BodyPosition(VectorizablePythonMsg):
    '''
    3D position in the body frame
    '''
    x1: float = field(default = 0.)
    x2: float = field(default = 0.)
    x3: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.x1, self.x2, self.x3])

    def from_vec(self, vec):
        self.x1, self.x2, self.x3 = vec

@dataclass
class ParametricPosition(VectorizablePythonMsg):
    ''' parametric position '''
    s: float = field(default = 0.)
    y: float = field(default = 0.)
    n: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.s, self.y, self.n])

    def from_vec(self, vec):
        self.s, self.y, self.n = vec

@dataclass
class OrientationQuaternion(VectorizablePythonMsg):
    ''' euler symmetric parameters '''
    qi: float = field(default = 0.)
    qj: float = field(default = 0.)
    qk: float = field(default = 0.)
    qr: float = field(default = 1.)

    def e1(self):
        '''
        longitudinal basis vector
        points in same direction the vehicle does
        '''
        return np.array([1 - 2*self.qj**2   - 2*self.qk**2,
                          2*(self.qi*self.qj + self.qk*self.qr),
                          2*(self.qi*self.qk - self.qj*self.qr)]).T

    def e2(self):
        '''
        lateral basis vector
        points to left side of vehicle from driver's perspective
        '''
        return np.array([2*(self.qi*self.qj - self.qk*self.qr),
                          1 - 2*self.qi**2   - 2*self.qk**2,
                          2*(self.qj*self.qk + self.qi*self.qr)]).T

    def e3(self):
        '''
        normal basis vector
        points towards top of vehicle
        '''
        return np.array([2*(self.qi*self.qk + self.qj*self.qr),
                          2*(self.qj*self.qk - self.qi*self.qr),
                          1 - 2*self.qi**2    - 2*self.qj**2]).T

    def R(self):
        # pylint: disable=line-too-long
        '''
        rotation matrix
        '''
        return np.array([[1 - 2*self.qj**2 - 2*self.qk**2,       2*(self.qi*self.qj - self.qk*self.qr), 2*(self.qi*self.qk + self.qj*self.qr)],
                         [2*(self.qi*self.qj + self.qk*self.qr), 1 - 2*self.qi**2 - 2*self.qk**2,       2*(self.qj*self.qk - self.qi*self.qr)],
                         [2*(self.qi*self.qk - self.qj*self.qr), 2*(self.qj*self.qk + self.qi*self.qr), 1 - 2*self.qi**2 - 2*self.qj**2      ]])

    def Rinv(self):
        # pylint: disable=line-too-long
        '''
        inverse rotation matrix
        '''
        return np.array([[1 - 2*self.qj**2 - 2*self.qk**2,       2*(self.qi*self.qj + self.qk*self.qr), 2*(self.qi*self.qk - self.qj*self.qr)],
                         [2*(self.qi*self.qj - self.qk*self.qr), 1 - 2*self.qi**2 - 2*self.qk**2,       2*(self.qj*self.qk + self.qi*self.qr)],
                         [2*(self.qi*self.qk + self.qj*self.qr), 2*(self.qj*self.qk - self.qi*self.qr), 1 - 2*self.qi**2 - 2*self.qj**2      ]])

    def norm(self):
        '''
        norm of the quaternion
        '''
        return np.sqrt(self.qr**2 + self.qi**2 + self.qj**2 + self.qk**2)

    def normalize(self):
        '''
        normalize a quaternion

        any orientation quaternion must always be normalized
        this function exists to help ensure that
        '''
        norm = self.norm()
        self.qr /= norm
        self.qi /= norm
        self.qj /= norm
        self.qk /= norm
        return

    def to_vec(self):
        return np.array([self.qi, self.qj, self.qk, self.qr])

    def from_vec(self, vec):
        self.qi, self.qj, self.qk, self.qr = vec

    def from_yaw(self, yaw):
        ''' quaternion from yaw (on a flat euclidean surface)'''
        self.qi = 0
        self.qj = 0
        self.qr = np.cos(yaw/2)
        self.qk = np.sin(yaw/2)

    def to_yaw(self):
        ''' quaternion to yaw (on a flat euclidean surface)'''
        return 2*np.arctan2(self.qk, self.qr)

    def from_mat(self, R):
        ''' update from a rotation matrix '''
        self.from_vec(transform.Rotation.from_matrix(R).as_quat())

    def qdot(self,w: 'BodyAngularVelocity') -> 'OrientationQuaternion':
        ''' derivative from body frame angular velocity '''
        qdot = OrientationQuaternion()
        qdot.qi =  0.5 * (self.qr * w.w1 + self.qj*w.w3 - self.qk*w.w2)
        qdot.qj =  0.5 * (self.qr * w.w2 + self.qk*w.w1 - self.qi*w.w3)
        qdot.qk =  0.5 * (self.qr * w.w3 + self.qi*w.w2 - self.qj*w.w1)
        qdot.qr = -0.5 * (self.qi * w.w1 + self.qj*w.w2 + self.qk*w.w3)
        return qdot

@dataclass
class EulerAngles(VectorizablePythonMsg):
    ''' euler angles '''
    a: float = field(default = 0.)
    ''' yaw angle '''
    b: float = field(default = 0.)
    ''' pitch angle '''
    c: float = field(default = 0.)
    ''' roll angle '''

    def to_vec(self):
        return np.array([self.a, self.b, self.c])

    def from_vec(self, vec):
        self.a, self.b, self.c = vec

    def R(self):
        ''' get rotation matrix '''
        a = self.a
        b = self.b
        c = self.c
        Ra = np.array([
            [np.cos(a),-np.sin(a),0],
            [np.sin(a), np.cos(a),0],
            [0        , 0        ,1]
        ])
        Rb = np.array([
            [np.cos(b),0, np.sin(b)],
            [0,        1, 0        ],
            [-np.sin(b),0, np.cos(b)]
        ])
        Rc = np.array([
            [1, 0,        0         ],
            [0, np.cos(c),-np.sin(c)],
            [0, np.sin(c), np.cos(c)]
        ])
        return Ra @ Rb @ Rc

    def from_mat(self, R):
        ''' update from a rotation matrix '''
        cba = transform.Rotation.from_matrix(R).as_euler('xyz', degrees=False)
        self.from_vec(np.flip(cba))

@dataclass
class Orientation(VectorizablePythonMsg):
    ''' some orientation measure in 3D '''

    @abstractmethod
    def R(self, v):
        ''' rotation matrix '''

    @abstractmethod
    def from_mat(self, R):
        ''' update from rotation matrix '''

@dataclass
class GlobalOrientation(Orientation):
    ''' global orientation '''

@dataclass
class RelativeOrientation(Orientation):
    ''' relative orientation'''

    @property
    @abstractmethod
    def ths(self) -> float:
        ''' get relative yaw angle estimate '''

    @abstractmethod
    def from_yaw(self, ths: float):
        ''' update orientation from only a yaw angle '''

@dataclass
class GlobalEulerAngles(EulerAngles, GlobalOrientation):
    ''' global euler angles '''

@dataclass
class RelativeEulerAngles(EulerAngles, RelativeOrientation):
    ''' relative euler angles '''

    @property
    def ths(self):
        return self.a

    def from_yaw(self, ths: float):
        self.a = ths

@dataclass
class GlobalQuaternion(OrientationQuaternion, GlobalOrientation):
    ''' global orientation quaternion'''

@dataclass
class RelativeQuaternion(OrientationQuaternion, RelativeOrientation):
    ''' relative orientation quaternion'''

    @property
    def ths(self):
        return self.to_yaw()

@dataclass
class BodyLinearVelocity(VectorizablePythonMsg):
    ''' body frame linear velocity '''
    v1: float = field(default = 0.)
    v2: float = field(default = 0.)
    v3: float = field(default = 0.)

    def mag(self):
        ''' magnitutde (speed) '''
        return np.sqrt(self.v1**2 + self.v2**2 + self.v3**2)

    def signed_mag(self):
        ''' magntitude, but negative if moving backwards '''
        return self.mag() * np.sign(self.v1)

    def to_vec(self):
        return np.array([self.v1, self.v2, self.v3])

    def from_vec(self, vec):
        self.v1, self.v2, self.v3 = vec

@dataclass
class BodyAngularVelocity(VectorizablePythonMsg):
    ''' body frame angular velocity '''
    w1: float = field(default = 0.)
    w2: float = field(default = 0.)
    w3: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.w1, self.w2, self.w3])

    def from_vec(self, vec):
        self.w1, self.w2, self.w3 = vec

@dataclass
class BodyLinearAcceleration(VectorizablePythonMsg):
    ''' body frame linear acceleration '''
    a1: float = field(default = 0.)
    a2: float = field(default = 0.)
    a3: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.a1, self.a2, self.a3])

    def from_vec(self, vec):
        self.a1, self.a2, self.a3 = vec

@dataclass
class BodyAngularAcceleration(VectorizablePythonMsg):
    ''' body frame angular acceleration'''
    a1: float = field(default = 0.)
    a2: float = field(default = 0.)
    a3: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.a1, self.a2, self.a3])

    def from_vec(self, vec):
        self.a1, self.a2, self.a3 = vec

@dataclass
class BodyForce(VectorizablePythonMsg):
    ''' a force in the body frame '''
    f1: float = field(default = 0.)
    f2: float = field(default = 0.)
    f3: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.f1, self.f2, self.f3])

    def from_vec(self, vec):
        self.f1, self.f2, self.f3 = vec

@dataclass
class BodyMoment(VectorizablePythonMsg):
    ''' a moment in the body frame '''
    k1: float = field(default = 0.)
    k2: float = field(default = 0.)
    k3: float = field(default = 0.)

    def to_vec(self):
        return np.array([self.k1, self.k2, self.k3])

    def from_vec(self, vec):
        self.k1, self.k2, self.k3 = vec

@dataclass
class BaseBodyState(NestedPythonMsg):
    ''' required fields for a body, ie. of a vehicle or point mass '''
    t: float = field(default = 0.)
    p: ParametricPosition = field(default = None)
    vb: BodyLinearVelocity = field(default = None)
    wb: BodyAngularVelocity = field(default = None)
    ab: BodyLinearAcceleration = field(default=None)
    x: Position = field(default = None)
    q: OrientationQuaternion = field(default = None)

@dataclass
class BaseTangentBodyState(BaseBodyState):
    ''' required fields for a body tangent to a surface '''
    ths: float = field(default = 0.)


class Renderable(ABC):
    '''
    an object that can be rendered in 3D using opengl
    '''
    @abstractmethod
    def draw(self):
        ''' draw the object'''


class Domain(ABC):
    '''
    base class for environments, ie. a surface,
    declares attributes for rendering
    '''
    periodic: bool = False
    ''' whether or not the domain is periodic '''
    view_center: np.ndarray
    ''' [x,y,z] center of the domain for viewing purposes '''
    view_scale: float
    ''' some length scale for how big the domain is for viewing purposes '''

    @abstractmethod
    def triangulate(self, ubo) -> Renderable:
        '''
        triangulate the environment and return a renderable for it
        ubo parameter is intended for a uniform buffer object wrapper class
        and may be mandatory for child classes
        '''

    @abstractmethod
    def camera_follow_mat(self, s: float = 0., y: float = 0.) -> np.ndarray:
        '''
        get a 3x3 orientation view matrix for following the tangent
        of a domain at coordinate s, y
        '''
