''' general purpose casadi untilities'''
from functools import singledispatch
from typing import Tuple, Dict, Any
import subprocess
import platform
import os
import shutil
import hashlib

import dill
import casadi as ca
import numpy as np

from vehicle_3d.utils.load_utils import get_project_folder

def ca_pos_abs(x, eps = 1e-3):
    '''
    smooth, positive apporoximation to abs(x)
    meant for tire slip ratios, where result must be nonzero
    '''
    return ca.sqrt(x**2 + eps**2)

def ca_abs(x):
    '''
    absolute value in casadi
    do not use for tire slip ratios
    used for quadratic drag: c * v * abs(v)
    '''
    return ca.if_else(x > 0, x, -x)

def ca_pos(x):
    '''
    max(x, 0) in casadi
    '''
    return ca.if_else(x > 0, x, 0)

def ca_smooth_pos(x, eps = 1e-3):
    '''
    max(x, 0) with smooth transition
    '''
    return x * ca_smooth_geq(x, eps)

def ca_leq(x, a: float = 1):
    '''
    min(x, a) in casadi
    '''
    return ca.if_else(x < a, x, a)

def ca_sign(x):
    ''' sign(x)'''
    return ca.if_else(x >= 0, 1, -1)

def ca_smooth_sign(x, eps = 1e-3):
    ''' smooth apporoximation to sign(x)'''
    return x / ca_pos_abs(x, eps)

def ca_smooth_geq(x, y = 0, eps = 1e-3):
    ''' smooth apporoximation to x >= y'''
    return 0.5 * ca_smooth_sign(x-y, eps) + 0.5

def ca_grad(y: ca.SX, x: ca.SX = None) -> ca.SX:
    '''
    gradient of a vector as in np.gradient
    i.e. y(x) sampled at the given points, with x strictly monotonic
    it is assumed that both x and y are of size (N,1)
    '''
    grad_y = []
    N = y.numel()
    if x is None:
        x = np.arange(N)
    for k in range(N):
        if k == 0:
            grad_y += [
                (y[k+1] - y[k]) / (x[k+1] - x[k])
            ]
        elif k == N-1:
            grad_y += [
                (y[k] - y[k-1]) / (x[k] - x[k-1])
            ]
        else:
            hd = x[k+1] - x[k]
            hs = x[k] - x[k-1]
            grad_y += [(y[k+1] - y[k])/hd/2 + (y[k] - y[k-1])/hs/2]
    grad_y = ca.vertcat(*grad_y)
    if hasattr(ca, 'cse'):
        grad_y = ca.cse(grad_y)
    return grad_y

def coerce_angle(th,):
    ''' take an angle th and coerce to interval [-pi/pi]'''
    th = ca.fmod(th, ca.pi*2)
    th = ca.if_else(th > np.pi, th - 2*ca.pi, th)
    th = ca.if_else(th < -np.pi, th + 2*ca.pi, th)
    return th

def coerce_angle_diff(th, th0 = 0):
    ''' take an angle th and coerce it to be within [-pi,pi] of th0 '''
    return th0 + coerce_angle(th - th0)

def hat(vec: ca.SX):
    ''' hat map '''
    return ca.vertcat(
        ca.horzcat(0, -vec[2], vec[1]),
        ca.horzcat(vec[2], 0, -vec[0]),
        ca.horzcat(-vec[1], vec[0], 0)
    )

def ca_function_dispatcher(func: ca.Function):
    '''
    utility for wrapping a casadi function in a new function
    which returns a symbolic expression if called symbolically
    and a numpy array otherwise.
    '''

    if func.n_out() == 1:
        def f(*args):
            return np.array(func(*args)).squeeze()

    else:
        def f(*args):
            return (np.array(output).squeeze() for output in func(*args))

    def f_ca(*args):
        return func(*args)

    f = singledispatch(f)
    f.register(ca.SX, f_ca)
    f.register(ca.MX, f_ca)

    return f

def check_free_sx(expr, args) -> Tuple[ca.Function, ca.SX]:
    '''
    check if a function with arguments args for expression expr
    can be made without free variables,
    returns the function if so and []
    otherwise returns (None, [free variables])
    '''
    try:
        func = ca.Function('f', args, expr)
    except RuntimeError:
        # casadi 3.6+ compat
        func = ca.Function('f', args, expr,{'allow_free':True})
    if len(func.free_sx()) == 0:
        return func, []
    return None, func.free_sx()

def _get_win_compiler() -> str:
    # pylint: disable=line-too-long
    # compiler list adapted from https://github.com/openhome/ohdevtools/blob/master/ci_build.py#L27
    compilers =  [
        'C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat',      # VS2022 Pro
        'C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat',         # VS2022 Community edition
        'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat',   # VS2019 Community edition
        'C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat',  # Build Tools for VS2017
        'C:\\Program Files\\Microsoft Visual Studio\\2017\\Professional\\Common7\\Tools\\vcvars64.bat',            # VS2017 Pro
        'C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat',   # VS2017 Community edition
        'C:\\Program Files (x86)\\Microsoft Visual Studio 12.0\\Common7\\Tools\\vsvars64.bat'                      # VS12 Express
    ]
    for vcvars64 in compilers:
        if os.path.exists(vcvars64):
            return vcvars64
    return None

def compiler_available() -> bool:
    ''' check if a compiler is available for the current platform '''
    if platform.system() == 'Linux':
        result = subprocess.run('which gcc', capture_output=True, shell=True, check=False)
        return result.stdout != b''
    elif platform.system() == 'Windows':
        return _get_win_compiler() is not None
    return False

def _get_compiled_solver_path(prob: Any):
    codegen_folder = os.path.join(get_project_folder(), 'build')
    if not os.path.exists(codegen_folder):
        os.mkdir(codegen_folder)
    solver_hash = hashlib.sha256()
    solver_hash.update(dill.dumps(prob))
    solver_id = 'solver_' + solver_hash.hexdigest()
    return codegen_folder, solver_id

def compile_solver(name: str, solver: str, prob: Dict[str, Any], opts: Dict[str, Any]) -> ca.nlpsol:
    ''' compile a casadi nlpsol object '''
    # pylint: disable=line-too-long
    if not compiler_available():
        raise RuntimeError('Unable to compile solver, no compiler is available')

    dest_folder, code_id = _get_compiled_solver_path(prob)
    codegen_fname = code_id + '.c'
    source_path = os.path.join(dest_folder, codegen_fname)

    if platform.system() == 'Linux':
        dest_path = os.path.join(dest_folder, code_id) + '.so'
        cmd = f'gcc -fPIC -shared -O3 {source_path} -o {dest_path}'

    elif platform.system() == 'Windows':
        vcvars64 = _get_win_compiler()
        dest_path = os.path.join(dest_folder, code_id)
        cmd = f'("{vcvars64}" x64>nul)&& cl /LD /Fo"{dest_path}.obj" /Fe"{dest_path}.dll" "{source_path}"'
        dest_path += '.dll'
    else:
        raise NotImplementedError(f'unsupported build platform {platform.system()}')

    if not os.path.exists(dest_path):
        nlpsol = ca.nlpsol(name, solver, prob, opts)
        nlpsol.generate_dependencies(codegen_fname)
        shutil.move(codegen_fname, source_path)
        subp = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        print(f'Compiling solver {name}... ')
        subp.wait()

    compiled_nlpsol = ca.nlpsol(name, solver, dest_path, opts)
    return compiled_nlpsol
