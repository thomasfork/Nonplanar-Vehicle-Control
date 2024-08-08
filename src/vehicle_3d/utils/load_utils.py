''' utilities for loading assets '''
import os

def get_project_folder() -> str:
    ''' root directory of the project '''
    project_folder = __file__
    for _ in range(4):
        project_folder = os.path.dirname(project_folder)
    return project_folder

def get_assets_folder() -> str:
    ''' return os path to assets folder for this module'''
    return os.path.join(get_project_folder(), 'assets')

def get_assets_file(file: str) -> str:
    ''' return os path to assets file for this module'''
    return os.path.join(get_assets_folder(), file)
