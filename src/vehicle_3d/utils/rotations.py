''' rotation parameterizations with symbolic dynamics support '''

from enum import Enum

import numpy as np
import casadi as ca

from vehicle_3d.pytypes import GlobalQuaternion, GlobalEulerAngles, RelativeQuaternion, \
    RelativeEulerAngles, Orientation

class Reference(Enum):
    '''
    reference frame for orientation
    '''
    GLOBAL = 0
    PARAMETRIC = 1


class Parameterization(Enum):
    '''
    rotation parameterization
    '''
    ESP = 0 # euler symmetric parameters (quaternion)
    YPR = 1 # yaw pitch roll (Tait-Bryan angles)


class Rotation:
    '''
    rotation handler
    '''
    ref: Reference
    param: Parameterization
    _sym_vars: ca.SX
    def __init__(self, ref: Reference, param: Parameterization):
        self.ref = ref
        self.param = param
        self._setup()

    def _setup(self):
        self._sym_vars = {}
        r = None
        R = None
        M = None
        if self.param == Parameterization.ESP:
            qi = ca.SX.sym('qi')
            qj = ca.SX.sym('qj')
            qk = ca.SX.sym('qk')
            qr = ca.SX.sym('qr')
            r = ca.vertcat(qi, qj, qk, qr)
            R = ca.vertcat(
                ca.horzcat(
                    1 - 2*qj**2 - 2*qk**2,
                    2*(qi*qj - qk*qr),
                    2*(qi*qk + qj*qr)
                ),
                ca.horzcat(
                    2*(qi*qj + qk*qr),
                    1 - 2*qi**2 - 2*qk**2,
                    2*(qj*qk - qi*qr),
                ),
                ca.horzcat(
                    2*(qi*qk - qj*qr),
                    2*(qj*qk + qi*qr),
                    1 - 2*qi**2 - 2*qj**2
                )
            ) / (qi**2 + qj**2 + qk**2 + qr**2)
            yaw = 2*ca.arctan2(qk, qr)
            M = 0.5 * ca.vertcat(
                ca.horzcat(
                    qr, -qk, qj
                ),
                ca.horzcat(
                    qk, qr, -qi
                ),
                ca.horzcat(
                    -qj, qi, qr
                ),
                ca.horzcat(
                    -qi, -qj, -qk
                )
            )
        elif self.param == Parameterization.YPR:
            a = ca.SX.sym('a')
            b = ca.SX.sym('b')
            c = ca.SX.sym('c')
            r = ca.vertcat(a, b, c)
            #rotation matrix for centerline orientation
            Ra = ca.vertcat( ca.horzcat(ca.cos(a),-ca.sin(a),0),
                            ca.horzcat(ca.sin(a), ca.cos(a),0),
                            ca.horzcat(0        , 0        ,1))
            Rb = ca.vertcat( ca.horzcat(ca.cos(b),0,ca.sin(b)),
                            ca.horzcat(0,        1, 0        ),
                            ca.horzcat(-ca.sin(b),0, ca.cos(b)))
            Rc = ca.vertcat( ca.horzcat(1, 0,        0         ),
                            ca.horzcat(0, ca.cos(c),-ca.sin(c)),
                            ca.horzcat(0, ca.sin(c), ca.cos(c)))
            R = Ra @ Rb @ Rc
            yaw = a

            M = ca.vertcat(
                ca.horzcat(0, ca.sin(c)/ca.cos(b), ca.cos(c)/ca.cos(b)),
                ca.horzcat(0, ca.cos(c), -ca.sin(c)),
                ca.horzcat(1, ca.sin(c)*ca.tan(b), ca.cos(c) * ca.tan(b))
            )
        else:
            raise NotImplementedError()

        assert r is not None
        assert yaw is not None
        assert R is not None
        assert M is not None

        self._sym_vars['r'] = r
        self._sym_vars['yaw'] = yaw
        self._sym_vars['R'] = R
        self._sym_vars['M'] = M

    def r(self):
        ''' rotation variables '''
        return self._sym_vars['r']

    def yaw_angle(self):
        ''' yaw angle of rotation '''
        return self._sym_vars['yaw']

    def R(self):
        ''' rotation matrix '''
        return self._sym_vars['R']

    def M(self):
        '''
        kinematics
        ie. d/dt(r) = M @ w
        w = [w1, w2, w3]
        '''
        return self._sym_vars['M']

    def ubr(self):
        '''
        upper bound on self.r()
        norm of ESP must be handled elsewhere in optimization
        '''
        if self.param == Parameterization.ESP:
            return [np.inf, np.inf, np.inf, np.inf]
        elif self.param == Parameterization.YPR:
            if self.ref == Reference.GLOBAL:
                return [np.inf, np.pi/2.1, np.pi/2.1]
            elif self.ref == Reference.PARAMETRIC:
                return [np.pi/2, np.pi/2.1, np.pi/2.1]
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def lbr(self):
        '''
        lower bound on self.r()
        norm of ESP must be handled elsewhere in optimization
        '''
        if self.param == Parameterization.ESP:
            return [-np.inf, -np.inf, -np.inf, -np.inf]
        elif self.param == Parameterization.YPR:
            if self.ref == Reference.GLOBAL:
                return [-np.inf, -np.pi/2.1, -np.pi/2.1]
            elif self.ref == Reference.PARAMETRIC:
                return [-np.pi/2, -np.pi/2.1, -np.pi/2.1]
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def get_empty_state(self) -> Orientation:
        '''
        get an empty state object for the orientation used
        '''
        if self.ref == Reference.GLOBAL:
            if self.param == Parameterization.ESP:
                s = GlobalQuaternion()
            elif self.param == Parameterization.YPR:
                s = GlobalEulerAngles()
            else:
                raise NotImplementedError()
        elif self.ref == Reference.PARAMETRIC:
            if self.param == Parameterization.ESP:
                s = RelativeQuaternion()
            elif self.param == Parameterization.YPR:
                s = RelativeEulerAngles()
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return s
