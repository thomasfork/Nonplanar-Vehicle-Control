'''
utilities for getting ipopt solver
'''
import os
from typing import Dict

import casadi as ca

def ipopt_solver(prob: Dict, opts: Dict = None,
        verbose: bool = False,
        name: str = 'solver',
        max_itr: int = 1000)\
         -> ca.nlpsol:
    ''' helper function to get ipopt solver instance '''
    if opts is None:
        opts = {}
    if not verbose:
        opts['ipopt.print_level'] = 0
        opts['print_time'] = 0
        opts['ipopt.sb'] = 'yes'
    elif '3.6' in ca.__version__:
        opts['ipopt.timing_statistics'] = 'yes'

    if os.path.exists('/usr/local/lib'):
        if 'libcoinhsl.so' in os.listdir('/usr/local/lib/') and \
                '3.6' in ca.__version__:
            # hsllib option is only supported on newer ipopt versions
            opts['ipopt.linear_solver'] = 'ma97'
            opts['ipopt.hsllib'] = '/usr/local/lib/libcoinhsl.so'
        elif 'libhsl.so' in os.listdir('/usr/local/lib/'):
            # check for obsolete hsl install and that it is on search path
            if '/usr/local/lib' in os.environ['LD_LIBRARY_PATH']:
                opts['ipopt.linear_solver'] = 'ma97'
            else:
                print('ERROR - HSL is present but not on path')
                print('Run "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/"')
                print('Defaulting to MUMPS')

    opts['ipopt.max_iter'] = max_itr

    return ca.nlpsol(name, 'ipopt', prob, opts)
