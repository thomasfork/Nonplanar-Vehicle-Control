'''
This python library implements nonplanar vehicle dynamics research
major dependencies are installed by this, a few that are left out:
glumpy: needed for legacy visualization scheme
'''
from setuptools import setup

setup(
    name='barc3d',
    version='0.1',
    packages=['barc3d'],
    install_requires=['numpy >= 1.19.5',
                      'matplotlib>=3.1.2',
                      'scipy>=1.4.1',
                      'sympy>=1.6.2',
                      'casadi>=3.5.1',
                      'imgui',
                      'numpy-stl',
                      'glfw',
                      'PyOpenGL']
)
