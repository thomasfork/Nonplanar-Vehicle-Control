from dataclasses import dataclass, field
from barc3d.pytypes import PythonMsg, VehicleState
from barc3d.control.base_controller import BaseController


@dataclass
class PIDConfig(PythonMsg):
    kp_l :float = field(default =    .1)  # longitudinal coefficients (acceleration)
    ki_l :float = field(default =    0.15)
    kd_l :float = field(default =    0.0)
    kp_t :float = field(default =    1.0) # transverse coefficients (steering)
    ki_t :float = field(default =    0.0)
    kd_t :float = field(default =    0.0)
    la:   float = field(default =    3) # look ahead term - angular error
    dt   :float = field(default =    0.1)

    vref: float = field(default =    10)  # target speed
    yref: float = field(default =    0)  # target centerline offset
    
class PID():
    def __init__(self, kp, ki, kd, dt):
        self.ki = ki
        self.kd = kd
        self.kp = kp
        self.dt = dt
        
        self.last_e = 0
        self.cum_e = 0
        self.ref = 0
        return
        
    def set_ref(self, ref):
        self.ref = ref
        return
    
    def step(self, x):
        e = self.ref - x
        self.cum_e += e * self.dt
        der_e = (self.last_e - e) / self.dt
        
        u = self.kp * e + self.ki * self.cum_e + self.kd * der_e
        
        self.last_e = e
        return u

class PIDController(BaseController):
    def __init__(self, config:PIDConfig = PIDConfig()):
        self.config = config
        self.long_controller = PID(config.kp_l, config.ki_l, config.kd_l, config.dt)
        self.tran_controller = PID(config.kp_t, config.ki_t, config.kd_t, config.dt)
        
        self.long_controller.set_ref(config.vref)
        self.tran_controller.set_ref(self.config.yref)
        
        return
    
    def step(self, state:VehicleState):
        a = self.long_controller.step(state.v.v1)
        y = self.tran_controller.step(state.p.y + state.p.ths * self.config.la )  
        
        a += state.q.e1()[2].__float__() * 9.81  # added term for current slope 
        
        
        state.u.a = a
        state.u.y = y
        return
