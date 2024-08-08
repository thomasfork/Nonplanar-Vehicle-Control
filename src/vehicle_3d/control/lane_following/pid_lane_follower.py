'''
PID control lane follower
one PID loop for longitudinal and one for lateral control
'''

from dataclasses import dataclass, field
from typing import Optional

from vehicle_3d.pytypes import PythonMsg, BaseTangentBodyState
from vehicle_3d.control.base_controller import Controller

@dataclass
class PIDConfig(PythonMsg):
    ''' PID lane follower configuration '''
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

class PID:
    ''' helper class for a single PID loop '''
    def __init__(self, kp, ki, kd, dt):
        self.ki = ki
        self.kd = kd
        self.kp = kp
        self.dt = dt

        self.last_e = 0
        self.cum_e = 0
        self.ref = 0

    def set_ref(self, ref):
        ''' set the PID controller reference '''
        self.ref = ref

    def step(self, x) -> float:
        ''' step the PID controller, returning a new input '''
        e = self.ref - x
        self.cum_e += e * self.dt
        der_e = (self.last_e - e) / self.dt

        u = self.kp * e + self.ki * self.cum_e + self.kd * der_e

        self.last_e = e
        return u

class PIDController(Controller):
    ''' PID lane follower '''
    def __init__(self, config: Optional[PIDConfig] = None):
        if config is None:
            config = PIDConfig()
        self.config = config
        self.long_controller = PID(config.kp_l, config.ki_l, config.kd_l, config.dt)
        self.lat_controller = PID(config.kp_t, config.ki_t, config.kd_t, config.dt)

        self.long_controller.set_ref(config.vref)
        self.lat_controller.set_ref(self.config.yref)

    def step(self, state:BaseTangentBodyState):
        a = self.long_controller.step(state.vb.v1)
        y = self.lat_controller.step(state.p.y + state.ths * self.config.la )

        a += float(state.q.e1()[2]) * 9.81  # added term for current slope

        state.u.a = a
        state.u.y = y
