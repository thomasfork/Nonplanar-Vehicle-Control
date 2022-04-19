import os
import numpy as np

from barc3d.surfaces.frenet_surface import FrenetSurface
from barc3d.surfaces.tait_bryan_surface import TaitBryanAngleSurface
from barc3d.surfaces.arc_profile_centerline_surface import ArcProfileCenterlineSurface

def get_surface_save_folder():
    relpath = os.path.join(__file__, os.path.pardir, 'data')
    return os.path.abspath(relpath)
        
def get_available_surfaces():
    save_folder = get_surface_save_folder()
    return os.listdir(save_folder)
    
def surface_name_to_filename(surface_name):
    return os.path.join(get_surface_save_folder(), str(surface_name))
    
def load_surface(surface_name):
    filename = surface_name_to_filename(surface_name)
    if not filename.endswith('.npz'): filename += '.npz'
    surf_data = np.load(filename, allow_pickle = True)
    
    if 'surf_class' not in surf_data.keys():
        raise TypeError('Invalid data type loaded')
    
    if surf_data['surf_class'] == 'FrenetSurface':
        surf = FrenetSurface()
    elif surf_data['surf_class'] == 'TaitBryanAngleSurface':
        surf = TaitBryanAngleSurface()
    elif surf_data['surf_class'] == 'ArcProfileCenterlineSurface':
        surf = ArcProfileCenterlineSurface()
    else:
        print('Unrecognized surface class %s'%surf_data['surf_class'])  
        surf = None
    
    if surf is not None:
        if surf.is_class_data(surf_data):
            surf.unpack_loaded_data(surf_data)
            surf.initialize(surf.config)
        else:
            print('Warning - data file matched innapropriately, this is probably a code issue')
    return surf
