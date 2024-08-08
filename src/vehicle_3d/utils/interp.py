'''
interpolation functions that support casadi evaluation
especially geared towards SX interpolation,
as opposed to built-in ca.interpolant
'''
from enum import Enum

import numpy as np
import casadi as ca
import scipy.interpolate


def linear_interpolant(x_data, y_data) -> ca.Function:
    '''
    univariate linear interpolation,
    out-of bounds data is clipped to last value
    '''
    x_interp = ca.SX.sym('x')
    y_lin = ca.pw_lin(x_interp, [*x_data, x_data[-1] + 1], [*y_data, y_data[-1]])
    return ca.Function('y', [x_interp], [y_lin])

def pchip_interpolant(x_data, y_data, extrapolate = 'const') -> ca.Function:
    '''
    univariate PCHIP interpolation
    close to linear interpolation but more derivatives are defined
    useful for approximating linear interpolation with more derivatives present
    '''
    spline = scipy.interpolate.PchipInterpolator(x_data, y_data)
    return scipy_spline_to_casadi(spline, extrapolate)

def spline_interpolant(x_data, y_data,
                bc_type = 'periodic',
                extrapolate = 'const',
                fast = False) -> ca.Function:
    '''
    univariate cubic spline interpolation
    for full documentation see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    main useful boundary conditions are:
    periodic
    not-a-knot
    clamped

    can optionally return a fast evaluation spline that is MX only

    It is possible to package a scipy spline as a casadi function
    however this does not result in any
    speed up because the function will still be evaluated 1 by 1
    by casadi.

    This returns a spline that can be used in CasADi's framework, but for
    fastest runtime one should use the scipy spline in a vectorized manner
    or other spline libraries.
    '''
    spline = scipy.interpolate.CubicSpline(x_data, y_data, bc_type = bc_type)
    if fast:
        return _fast_ca_spline(spline)
    return scipy_spline_to_casadi(spline, extrapolate)

def scipy_spline_to_casadi(spline, extrapolate = 'const'):
    '''
    converts a scipy spline function to a casadi function

    by default, truncates to the last spline value, however it can extrapolate linearly as well,
    which is useful when fitting a spline to a centerline (ie. spline_surface.py)
    '''
    k_c = spline.c
    k_x = spline.x
    k_f = spline(k_x[-1])

    x_fit = ca.SX.sym('s')
    x_0 = ca.pw_const(x_fit, k_x, [k_x[0], *k_x])
    c_0 = ca.pw_const(x_fit, k_x, [k_c[3,0], *k_c[3,:], k_f])
    if extrapolate == 'const':
        c_1 = ca.pw_const(x_fit, k_x, [0,        *k_c[2,:], 0])
    elif extrapolate == 'linear':
        c_1 = ca.pw_const(x_fit, k_x, [k_c[2,0], *k_c[2,:], spline(k_x[-1],1)])
    else:
        raise NotImplementedError(f'Unrecognized extrapolation key: {extrapolate}')


    c_2 = ca.pw_const(x_fit, k_x, [0,        *k_c[1,:], 0])
    c_3 = ca.pw_const(x_fit, k_x, [0,        *k_c[0,:], 0])

    x_rel = x_fit - x_0
    y_fit = c_0 + c_1*x_rel + c_2*x_rel**2 + c_3*x_rel**3
    f_y = ca.Function('k', [x_fit], [y_fit])

    return f_y

def _fast_ca_spline(spline):
    '''
    a spline that evaluates faster for large numbers of knots
    but is MX only
    particularly useful for long racetracks
    with problems that fix path length during optimization

    this is slightly slower at runtime than ca.interpolant('', 'bspline',...)
    but allows the added flexibility of scipy's spline fitting.

    Due to the use of ca.low this should not be differentiated,
    the first three derivatives are returned to avoid the need for this
    '''
    k_c = spline.c
    k_x = spline.x

    s = ca.MX.sym('s')
    sc = ca.if_else(s < k_x.min(), k_x.min(), ca.if_else(s > k_x.max(), k_x.max(), s))
    idx = ca.low(k_x,s)


    k_x = ca.MX(ca.DM(k_x))
    k_c = ca.MX(ca.DM(k_c))

    offset = k_x[idx]
    coeffs = k_c[:,idx]

    s_rel = sc - offset

    # fits for spline and its derivatives
    s0 = s_rel**ca.DM(np.arange(3,-1,-1))
    s1 = s_rel**ca.DM(np.arange(2,-1,-1)) * ca.DM([3,2,1])
    s2 = s_rel**ca.DM(np.arange(1,-1,-1)) * ca.DM([6,2])
    s3 = ca.DM([6])

    fit0 = ca.dot(coeffs, s0)
    fit1 = ca.dot(coeffs[:-1], s1)
    fit2 = ca.dot(coeffs[:-2], s2)
    fit3 = ca.dot(coeffs[:-3], s3)

    f = ca.Function('x',[s],[fit0])
    f1 = ca.Function('x',[s],[fit1])
    f2 = ca.Function('x',[s],[fit2])
    f3 = ca.Function('x',[s],[fit3])

    return f, f1, f2, f3

class InterpMethod(Enum):
    ''' interpolation method for 1D scalar data '''
    LINEAR = linear_interpolant
    PCHIP = pchip_interpolant
    SPLINE = spline_interpolant

def interp(x_data: np.ndarray, y_data: np.ndarray, method: InterpMethod,
           extrapolate: str = 'const',
           fast_mx_spline: bool = False,
           spline_bc: str = 'periodic'):
    ''' general purpose function to handle an interpolation method on 1D scalar data '''
    if all(y_data == 0):
        return ca.Function('zero',[ca.SX.sym('s')],[0])
    if method == InterpMethod.LINEAR:
        return linear_interpolant(x_data, y_data)
    elif method == InterpMethod.PCHIP:
        return pchip_interpolant(x_data, y_data,
                                 extrapolate = extrapolate)
    elif method == InterpMethod.SPLINE:
        return spline_interpolant(x_data, y_data,
                                  bc_type = spline_bc,
                                  fast = fast_mx_spline,
                                  extrapolate = extrapolate )
    else:
        raise NotImplementedError(f'Unhandled Key {method}')


class RBF_KERNELS(Enum):
    ''' enum of available kernels'''
    INVERSE_QUADRATIC = 'inverse_quadratic'
    GAUSSIAN = 'gaussian'


class RBF(scipy.interpolate.RBFInterpolator):
    '''
    radial basis function interpolation in 2D
    based off of the scipy.interpolant.RBFInterpolator class

    resulting interpolant is packaged into a casadi function
    such as for an elevation map surface
    '''
    def __init__(self, X, Y, Z, epsilon=1, kernel: RBF_KERNELS = RBF_KERNELS.GAUSSIAN):
        xy = np.array([np.concatenate(X), np.concatenate(Y)]).T
        z = np.concatenate(Z)
        super().__init__(xy, z, kernel=kernel.value, epsilon=epsilon, degree=1)
        self._get_kernel()
        self._setup(xy)

    def _get_kernel(self):
        '''
        obtains a kernel as a casadi function from a string'''

        r = ca.SX.sym('r')
        if self.kernel == RBF_KERNELS.GAUSSIAN.value:
            y = ca.exp(-r**2)
            self.f_kernel = ca.Function('k', [r], [y])
        elif self.kernel == RBF_KERNELS.INVERSE_QUADRATIC.value:
            y = 1/(1+r**2)
            self.f_kernel = ca.Function('k', [r], [y])
        else:
            raise NotImplementedError(f'unsupported kernel {self.kernel}')

    def _setup(self, xy_data):
        ''' builds casadi function from already fit RBF '''
        x = ca.SX.sym('x', 2)

        x_shift_scale = (x - self._shift) / self._scale

        z = self._coeffs[-3] + x_shift_scale.T @ self._coeffs[-2:]

        for k, xy in enumerate(xy_data):
            dx = (x[0] - xy[0]) * self.epsilon
            dy = (x[1] - xy[1]) * self.epsilon
            r = ca.sqrt(dx**2 + dy**2)

            z = z + self.f_kernel(r) * self._coeffs[k]

        dz = ca.jacobian(z, x)
        ddz = ca.jacobian(dz, x)

        self.f_z = ca.Function('z', [x], [z])
        self.f_dz = ca.Function('z', [x], [dz])
        self.f_ddz = ca.Function('z', [x], [ddz])

        self.f_z_np = lambda x: np.array(self.f_z(x)).squeeze()

    def get_rbf(self):
        ''' return RBF interplation object in CasADi'''
        return self.f_z

    def get_rbf_jac(self):
        ''' return function for jacobian '''
        return self.f_dz

    def get_rbf_hess(self):
        ''' return function for hessian '''
        return self.f_ddz

    def get_rbf_np(self):
        ''' return RBF interpolation object in numpy'''
        return self.f_z_np


class BivariateSpline(ca.Callback):
    '''
    bivariate scipy spline with casadi compatibility
    '''
    # pylint: disable=arguments-differ, unused-argument
    def __init__(self, name, **args):
        self.spline = scipy.interpolate.RectBivariateSpline(**args)
        self.fwd_callback: _BivariateSplineJacobian = None
        ca.Callback.__init__(self)
        self.construct(name)

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self,i):
        return ca.Sparsity.dense(2,1)

    def get_sparsity_out(self,i):
        return ca.Sparsity.dense(1,1)

    def eval(self, arg):
        return [ca.DM(self.spline(arg[0][0], arg[0][1]))]

    def has_forward(self, nfwd):
        return nfwd == 1

    def get_forward(self,nfwd,name,inames,onames,opts):
        self.fwd_callback = _BivariateSplineJacobian(self.spline, name, opts)
        return self.fwd_callback


class _BivariateSplineJacobian(ca.Callback):
    # pylint: disable=arguments-differ, unused-argument
    def __init__(self, spline: scipy.interpolate.BivariateSpline, name, opts=None):
        self.spline = spline
        self.fwd_callback: _BivariateSplineHessian = None
        ca.Callback.__init__(self)
        if opts is None:
            opts = {}
        self.construct(name, opts)

    def get_n_in(self):
        return 3

    def get_n_out(self):
        return 1

    def get_sparsity_in(self,i):
        if i==0: # nominal input
            return ca.Sparsity.dense(2,1)
        elif i==1: # nominal output
            return ca.Sparsity(1,1)
        else: # Forward seed
            return ca.Sparsity.dense(2,1)

    def get_sparsity_out(self,i):
        # Forward sensitivity
        return ca.Sparsity.dense(1,1)

    def eval(self, arg):
        if arg[2][0] != 0:
            ret = self.spline(arg[0][0], arg[0][1], dx = 1, dy = 0) * arg[2][0]
        if arg[2][1] != 0:
            ret = self.spline(arg[0][0], arg[0][1], dx = 0, dy = 1) * arg[2][1]
        return [ret]

    def has_forward(self, nfwd):
        return nfwd == 1

    def get_forward(self,nfwd,name,inames,onames,opts):
        self.fwd_callback = _BivariateSplineHessian(self.spline, name, opts)
        return self.fwd_callback


class _BivariateSplineHessian(ca.Callback):
    # pylint: disable=arguments-differ, unused-argument
    def __init__(self, spline: scipy.interpolate.BivariateSpline, name, opts=None):
        self.spline = spline
        ca.Callback.__init__(self)
        if opts is None:
            opts = {}
        self.construct(name, opts)

    def get_n_in(self):
        return 7

    def get_n_out(self):
        return 1

    def get_sparsity_in(self,i):
        if i==0: # nominal input
            return ca.Sparsity.dense(2,1)
        elif i==1: # nominal output
            return ca.Sparsity(1,1)
        elif i == 2: # Forward seed
            return ca.Sparsity.dense(2,1)
        elif i == 3:
            return ca.Sparsity.dense(1,1)
        elif i == 4:
            return ca.Sparsity.dense(2,1)
        elif i == 5:
            return ca.Sparsity.dense(1,1)
        return ca.Sparsity.dense(2,1)

    def get_sparsity_out(self,i):
        # Forward sensitivity
        return ca.Sparsity.dense(1,1)

    def eval(self, arg):
        dx = 0
        dy = 0
        sensitivity = 1
        if arg[2][0] != 0:
            dx += 1
            sensitivity *= arg[2][0]
        if arg[4][0] != 0:
            dx += 1
            sensitivity *= arg[4][0]
        if arg[2][1] != 0:
            dy += 1
            sensitivity *= arg[2][1]
        if arg[4][1] != 0:
            dy += 1
            sensitivity *= arg[4][1]
        return [self.spline(arg[0][0], arg[0][1], dx = dx, dy = dy) * sensitivity]
