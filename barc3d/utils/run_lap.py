
import sys
import pdb
import numpy as np
from numpy import pi as pi
import time as time

from barc3d.pytypes import VehicleState 



def run_solo_lap(controller, simulator, surf, figure = None, plot = False, lap = '', state = None, verbose = True, realtime = False):
    
    if state is None: state = VehicleState()
    t = 0
    n = 0
    t = 0
    n = 0
    done = False
    traj = []
    avg_dt = 0
    
    if figure is None and plot: plot = False

    while not done:
        ts = time.time()
        controller.step(state)
        
        simulator.step(state)
        
        if plot and figure is not None:
            done = not figure.draw(state)
            
        if state.p.s > surf.s_max(0)-5: done = True
        
        
        traj.append(state.copy())
        
        tf = time.time()
        avg_dt = n/(n+1) * avg_dt + 1/(n+1) * ((tf-ts))
        
        
        
        if verbose:
            if n > 0:
                sys.stdout.write("\033[F") # Cursor up one line
                sys.stdout.write("\033[K") # clear line
                sys.stdout.write("\033[F") # Cursor up one line
                sys.stdout.write("\033[K") # clear line
                sys.stdout.write("\033[F") # Cursor up one line
                sys.stdout.write("\033[K") # clear line
            print('Lap %s   Contact: %s'%(lap, 'OK  ' if state.fb.f3 >0 else 'LOST'))
            print('Vehicle @ %8.2f/%8.2f'%(state.p.s, surf.s_max(0)-5))
            print('Running @ %8.2fHz (avg: %8.2fHz)'%(1/(tf-ts), 1/avg_dt))    #NOTE: the frequency here is for the whole simulation, not just the controller solve time. 
        
        
        if realtime:
            if time.time() - ts < simulator.dt:
                time.sleep(simulator.dt - (time.time() - ts))
        else:
            time.sleep(0.001)     
        n += 1  
    return traj
