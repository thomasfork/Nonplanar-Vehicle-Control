import casadi as ca
import numpy as np
from barc3d.utils.collocation_utils import get_collocation_coefficients

def ca_abs(x, eps = 1e-9):
    return ca.sqrt(x**2 + eps**2)

def ca_sign(x, eps = 1e-9):
    return x / ca_abs(x)
    
def unpack_solution_helper(label, arg, out):
    f = ca.Function(label, [arg], out)
    return lambda arg: np.array(f(arg)).squeeze()
       

