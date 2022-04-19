'''
Standard types for nonplanar vehicle dynamics, ie. vehicle state and input
'''
from dataclasses import dataclass, field
import numpy as np
import copy
import sys

import casadi as ca
from barc3d.utils.ca_utils import ca_abs


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
            raise TypeError ('Cannot add new field "%s" to frozen class %s' %(key,self))
        else:
            object.__setattr__(self,key,value)


    def print(self, depth = 0, name = None):
        '''
        default __str__ method is not easy to read, especially for nested classes.
        This is easier to read but much longer

        '''
        print_str = ''
        for _ in range(depth):
            print_str += '  '
        if name:
            print_str += name + ' (' + type(self).__name__ + '):\n'
        else:
            print_str += type(self).__name__ + ':\n'
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                print_str += val.print(depth = depth + 1, name = key)
            else:
                for _ in range(depth + 1):
                    print_str += '  '
                print_str += str(key) + '=' + str(val)
                print_str += '\n'

        if depth == 0:
            print(print_str)
        else:
            return print_str

    def copy(self):
        ''' creates a copy of this class instance'''
        return copy.deepcopy(self)

@dataclass
class TireConfig(PythonMsg):
    ''' configuration class for a Pacjeka tire'''
    Bx: float = field(default = 16)
    Cx: float = field(default = 1.58)
    Ex: float = field(default = 0.1)
    By: float = field(default = 13)
    Cy: float = field(default = 1.45)
    Ey: float = field(default = -0.8)

    Exa: float = field(default = -0.5)
    Cxa: float = field(default = 1)
    rBx1: float = field(default = 13)
    rBx2: float = field(default = 9.7)

    Eys: float = field(default = 0.3)
    Cys: float = field(default = 1)
    rBy1: float = field(default = 10.62)
    rBy2: float = field(default = 7.82)

    mu:float = field(default = 0.75)
    r:  float = field(default =    0.3)     # tire geometric radius to ground
    re: float = field(default =    0.3)    # tire effective radius of turning

    I: float = field(default = 5) # tire moment of inertia that opposes angular acceleration

    # slip input and input rate limits (for slip input models)
    s_max: float = field(default = 1)
    s_min: float = field(default =-1)
    ds_max: float = field(default = 2)
    ds_min: float = field(default =-2)
    
    def get_throttle_map(self, g = 9.81):
        sigma = ca.SX.sym('s')
        ax, _ = self.combined_slip_forces(sigma, 0, g)
        f_ax = ca.Function('ax', [sigma], [ax])
        
        s = np.linspace(0,1,1000)
        a = np.array(f_ax(s)).squeeze()
        I = np.argmax(a)
        
        f_s = lambda target_accel: np.interp(target_accel, a[:I], s[:I]) if target_accel >= 0 else -np.interp(-target_accel, a[:I], s[:I])
        
        f_a = lambda target_sigma: np.array(f_ax(target_sigma)).squeeze()
        
        return f_a, f_s
    
    def combined_slip_forces(self, s,a,N):
        '''
        computes combined slip lateral/longitudinal tire forces using
            s: slip ratio
            a: slip angle
            N: normal force
        '''
        Fx0 = self.mu * N * ca.sin(self.Cx * ca.arctan((1 - self.Ex) * self.Bx * s +
                                                       self.Ex * ca.arctan(self.Bx * s)))
        Fy0 = self.mu * N * ca.sin(self.Cy * ca.arctan((1 - self.Ey) * self.By * a +
                                                       self.Ey * ca.arctan(self.By * a)))

        Bxa = self.rBx1 * ca.cos(ca.arctan(self.rBx2 * s))
        Gxa = ca.cos(self.Cxa * ca.arctan(Bxa * a - self.Exa * (Bxa * a - ca.arctan(Bxa * a))))

        Bys = self.rBy1 * ca.cos(ca.arctan(self.rBy2 * a))
        Gys = ca.cos(self.Cys * ca.arctan(Bys * s - self.Eys * (Bys * s - ca.arctan(Bys * s))))

        Fx  = Fx0 * Gxa
        Fy  = Fy0 * Gys
        return Fx,Fy
    
    def _sigma_alpha(self, w, y, v1,v2,v3,w1,w2,w3,x1,x2,x3):
    
        # body frame velocity at road surface
        v1_t  = v1 - w3*x2 + w2 * (x3 + self.r)
        v2_t  = v2 + x1*w3 - w1 * (x3 + self.r)

        # body frame velocity at effective radius
        vr1_t = v1 - w3*x2 + w2 * (x3 + self.re)
        vr2_t = v2 + w3*x1 - w1 * (x3 + self.re)

        # tire frame velocity at road surface
        vl_t  = v1_t * ca.cos(y) + v2_t * ca.sin(y)
        vt_t  = v2_t * ca.cos(y) - v1_t * ca.sin(y)

        # tire frame velocity at effective radius
        vrl_t = vr1_t * ca.cos(y) + vr2_t * ca.sin(y)
        vrt_t = vr2_t * ca.cos(y) - vr1_t * ca.sin(y)

        # slip angle
        alpha = -ca.arctan(vt_t / ca_abs(vrl_t))

        # slip ratio
        sigma = - (vrl_t - w * self.re) / ca_abs(vrl_t)
        
        return sigma, alpha

    def tire_model(self, w,y,T,N,v1,v2,v3,w1,w2,w3,x1,x2,x3):
        '''
        helper function to
        1. translate velocity (v1,v2) centered on the tire into the tire frame
        2. compute slip angle and slip ratio
        3. compute longitudinal and lateral tire force
        4. rotate forces back into body frame
        5. compute angular acceleration of the tire (due to slip and actuation torque T)

        arguments:
        w: angular velocity of the tire
        y: steering angle
        T: torque on the wheel (brakes + engine)
        N: normal force
        v1, v2, v3: body frame linear  velocity at vehicle COM
        w1, w2, w3: body frame angular velocity at vehicle COM
        x1, x2, x3: position of tire center in body frame, relative to COM
        '''
        
        sigma, alpha = self._sigma_alpha(w, y, v1,v2,v3,w1,w2,w3,x1,x2,x3)

        # forces
        Fl, Ft = self.combined_slip_forces(sigma, alpha, N)
        F1 = Fl * ca.cos(y) - Ft * ca.sin(y)
        F2 = Fl * ca.sin(y) + Ft * ca.cos(y)

        # angular acceleration
        w_dot = (-Fl * self.re + T) / self.I

        return sigma, alpha, F1, F2, w_dot

    def tire_model_sigma_input(self, sigma,y,N,v1,v2,v3,w1,w2,w3,x1,x2,x3):

        _, alpha = self._sigma_alpha(0, y, v1,v2,v3,w1,w2,w3,x1,x2,x3)

        # forces
        Fl, Ft = self.combined_slip_forces(sigma, alpha, N)
        
        F1 = Fl * ca.cos(y) - Ft * ca.sin(y)
        F2 = Fl * ca.sin(y) + Ft * ca.cos(y)

        return alpha, F1, F2

    def tire_model_lateral(self, y, N, v1, v2, v3, w1, w2, w3, x1, x2, x3):
        '''
        same as full tire model, but only for slip angle, and with pure side slip assumption
        '''
        _, alpha = self._sigma_alpha(0, y, v1,v2,v3,w1,w2,w3,x1,x2,x3)

        _, Fy0 = self.combined_slip_forces(0, alpha, N)
        
        F1 = -Fy0 * ca.sin(y)
        F2 = Fy0 * ca.cos(y)

        return alpha, F1, F2
        

@dataclass
class VehicleConfig(PythonMsg):
    dt :float = field(default =    0.10)    # simulation interval
    m  :float = field(default = 2303.0)     # vehicle mass in kg
    g  :float = field(default =    9.81)    # gravitational acceleration
    lf :float = field(default =    1.521)   # distance from COM to front axle
    lr :float = field(default =    1.499)   # distance from COM to rear axle
    tf :float = field(default =    0.625)   # disance from COM to left/right front wheels
    tr :float = field(default =    0.625)   # disance from COM to left/right rear wheels
    h  :float = field(default =    0.592)   # distance from road surface to COM

    bf : float = field(default =   1.8)     # distance from COM to front bumper
    br : float = field(default =   1.8)     # distance from COM to rear bumper

    I1: float = field(default =  955.9)     # roll inertia  (w1)
    I2: float = field(default = 5000.0)     # pitch inertia (w2) (estimated)
    I3: float = field(default = 5520.1)     # yaw inertia   (w3)

    c1: float = field(default =    0.0)     # longitudinal drag coefficient Fd1 = -c1 * v1 * abs(v1)
    c2: float = field(default =    0.0)     # transverse   drag coefficient Fd2 = -c2 * v2 * abs(v2)
    c3: float = field(default =    0.0)     # downforce    drag coefficient Fd3 = -c3 * v1**2

    a_max: float = field(default = 10)      # maximum acceleration
    a_min: float = field(default = -10)     # minimum acceleration
    y_max: float = field(default = 0.5)     # maximum steering angle
    y_min: float = field(default = -0.5)    # minimum steering angle

    da_max: float = field(default = 50)     # max acceleration rate of change
    da_min: float = field(default =-50)     # min acceleration rate of change
    dy_max: float = field(default = 2.5)
    dy_min: float = field(default =-2.5)

    N_max: float = field(default = 40000)

    tire: TireConfig = field(default = None)

    # in initial dynamics formulations, the parametric surface was not treated as the road surface
    # but one containing the vehicle center of mass.
    # For legacy support thereof, set the below to False
    # major effects when True:
    #  1. vehicle state p.n = h
    #  2. road surface texture is not offset by -h
    #  3. dynamics models factor in ca.inv(I - nII) rather than ca.inv(I)
    road_surface: bool = field(default = True)


    def __post_init__(self):
        if self.tire is None: self.tire = TireConfig()
    

@dataclass
class Position(PythonMsg):
    xi: float = field(default = 0)
    xj: float = field(default = 0)
    xk: float = field(default = 0)

    def xdot(self, q: 'OrientationQuaternion', v: 'BodyLinearVelocity') -> 'Position':
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
class BodyPosition(PythonMsg):
    '''
    position in the body frame
    recommended to be used for sensor locations
    vehicle is always at 0,0,0
    '''
    x1: float = field(default = 0)
    x2: float = field(default = 0)
    x3: float = field(default = 0)
    
    def to_vec(self):
        return np.array([self.x1, self.x2, self.x3])
        
    def from_vec(self, vec):
        self.x1, self.x2, self.x3 = vec

@dataclass
class BodyLinearVelocity(PythonMsg):
    v1: float = field(default = 0)
    v2: float = field(default = 0)
    v3: float = field(default = 0)

    def mag(self):
        return np.sqrt(self.v1**2 + self.v2**2 + self.v3**2)

    def signed_mag(self):
        return self.mag() * np.sign(self.v1)
@dataclass
class BodyAngularVelocity(PythonMsg):
    w1: float = field(default = 0)
    w2: float = field(default = 0)
    w3: float = field(default = 0)

@dataclass
class BodyLinearAcceleration(PythonMsg):
    a1: float = field(default = 0)
    a2: float = field(default = 0)
    a3: float = field(default = 0)

@dataclass
class BodyAngularAcceleration(PythonMsg):
    a1: float = field(default = 0)
    a2: float = field(default = 0)
    a3: float = field(default = 0)


@dataclass
class OrientationQuaternion(PythonMsg):
    qr: float = field(default = 1)
    qi: float = field(default = 0)
    qj: float = field(default = 0)
    qk: float = field(default = 0)

    def e1(self):
        return np.array([[1 - 2*self.qj**2   - 2*self.qk**2,
                          2*(self.qi*self.qj + self.qk*self.qr),
                          2*(self.qi*self.qk - self.qj*self.qr)]]).T

    def e2(self):
        return np.array([[2*(self.qi*self.qj - self.qk*self.qr),
                          1 - 2*self.qi**2   - 2*self.qk**2,
                          2*(self.qj*self.qk + self.qi*self.qr)]]).T

    def e3(self):
        return np.array([[2*(self.qi*self.qk + self.qj*self.qr),
                          2*(self.qj*self.qk - self.qi*self.qr),
                          1 - 2*self.qi**2    - 2*self.qj**2]]).T

    def R(self):
        return np.array([[1 - 2*self.qj**2 - 2*self.qk**2,       2*(self.qi*self.qj - self.qk*self.qr), 2*(self.qi*self.qk + self.qj*self.qr)],
                         [2*(self.qi*self.qj + self.qk*self.qr), 1 - 2*self.qi**2 - 2*self.qk**2,       2*(self.qj*self.qk - self.qi*self.qr)],
                         [2*(self.qi*self.qk - self.qj*self.qr), 2*(self.qj*self.qk + self.qi*self.qr), 1 - 2*self.qi**2 - 2*self.qj**2      ]])

    def Rinv(self):
        return np.array([[1 - 2*self.qj**2 - 2*self.qk**2,       2*(self.qi*self.qj + self.qk*self.qr), 2*(self.qi*self.qk - self.qj*self.qr)],
                         [2*(self.qi*self.qj - self.qk*self.qr), 1 - 2*self.qi**2 - 2*self.qk**2,       2*(self.qj*self.qk + self.qi*self.qr)],
                         [2*(self.qi*self.qk + self.qj*self.qr), 2*(self.qj*self.qk - self.qi*self.qr), 1 - 2*self.qi**2 - 2*self.qj**2      ]])

    def norm(self):
        return np.sqrt(self.qr**2 + self.qi**2 + self.qj**2 + self.qk**2)

    def normalize(self):
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
        self.qi = 0
        self.qj = 0
        self.qr = np.cos(yaw/2)
        self.qk = np.sin(yaw/2)
        return

    def to_yaw(self):
        return 2*np.arctan2(self.qk, self.qr)

    def qdot(self,w: BodyAngularVelocity) -> 'OrientationQuaternion':
        qdot = OrientationQuaternion()
        qdot.qr = -0.5 * (self.qi * w.w1 + self.qj*w.w2 + self.qk*w.w3)
        qdot.qi =  0.5 * (self.qr * w.w1 + self.qj*w.w3 - self.qk*w.w2)
        qdot.qj =  0.5 * (self.qr * w.w2 + self.qk*w.w1 - self.qi*w.w3)
        qdot.qk =  0.5 * (self.qr * w.w3 + self.qi*w.w2 - self.qj*w.w1)
        return qdot

@dataclass
class ParametricPose(PythonMsg):
    s: float = field(default = 0)
    y: float = field(default = 0)
    n: float = field(default = 0)
    ths: float = field(default = 0)

@dataclass
class ParametricVelocity(PythonMsg):
    ds: float = field(default = 0)
    dy: float = field(default = 0)
    dn: float = field(default = 0)
    dths: float = field(default = 0)


@dataclass
class VehicleActuation(PythonMsg):
    a: float = field(default = 0)  # longitudinal acceleration due to tire forces
    y: float = field(default = 0)  # steering angle of front tires

@dataclass
class BodyForce(PythonMsg):
    f1: float = field(default = 0)
    f2: float = field(default = 0)
    f3: float = field(default = 0)

@dataclass
class DriveState(PythonMsg):
    '''
    hardware state of the vehicle, ie. driveline and control units
    throttle steering and brake in units of the vehicle (ie. us pulse width)
    as well as wheel angular velocities
    with positive convention for the vehicle moving forwards
    '''
    throttle: float = field(default = 0)
    steering: float = field(default = 0)
    brake: float    = field(default = 0)
    wfr: float      = field(default = 0)
    wfl: float      = field(default = 0)
    wrr: float      = field(default = 0)
    wrl: float      = field(default = 0)

@dataclass
class TireState(PythonMsg):
    y: float = field(default = 0) # short for gamma - steering angle
    a: float = field(default = 0) # short for alpha - slip angle
    s: float = field(default = 0) # short for sigma - slip ratio
    N: float = field(default = 0) # normal force

@dataclass
class VehicleState(PythonMsg):
    t: float                    = field(default = 0)        # time in seconds
    x: Position                 = field(default = None)     # global position
    q: OrientationQuaternion    = field(default = None)     # global orientation
    v: BodyLinearVelocity       = field(default = None)     # body linear velocity
    w: BodyAngularVelocity      = field(default = None)     # body angular velocity
    a: BodyLinearAcceleration   = field(default = None)     # body linear acceleration
    aa: BodyAngularAcceleration = field(default = None)     # body angular acceleration

    u: VehicleActuation         = field(default = None)     # actuation
    du: VehicleActuation        = field(default = None)     # actuation rate

    p: ParametricPose           = field(default = None)     # parametric position (s,y, ths)
    pt:ParametricVelocity       = field(default = None)     # parametric velocity (ds, dy, dths)

    fb: BodyForce               = field(default = None)

    hw: DriveState              = field(default = None)

    t_sol:float                 = field(default = -1)   # intended for controller solve time

    tfr: TireState              = field(default = None)
    tfl: TireState              = field(default = None)
    trr: TireState              = field(default = None)
    trl: TireState              = field(default = None)

    def __post_init__(self):
        if self.x is None: self.x = Position()
        if self.q is None: self.q = OrientationQuaternion()
        if self.v is None: self.v = BodyLinearVelocity()
        if self.w is None: self.w = BodyAngularVelocity()
        if self.a is None: self.a = BodyLinearAcceleration()
        if self.aa is None: self.aa = BodyAngularAcceleration()

        if self.u is None: self.u = VehicleActuation()
        if self.du is None: self.du = VehicleActuation()

        if self.p is None: self.p = ParametricPose()
        if self.pt is None:self.pt = ParametricVelocity()

        if self.fb is None: self.fb = BodyForce()

        if self.hw is None: self.hw = DriveState()

        if self.tfr is None: self.tfr = TireState()
        if self.tfl is None: self.tfl = TireState()
        if self.trr is None: self.trr = TireState()
        if self.trl is None: self.trl = TireState()
        return

