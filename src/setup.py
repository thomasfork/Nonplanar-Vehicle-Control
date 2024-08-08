'''
code for nonplanar vehicle modeling and control
'''
from setuptools import setup

extras = {
   'with_recording': ['opencv-python']
}

setup(
    packages=['vehicle_3d'],
    install_requires=[
        'numpy',
        'casadi',
        'imgui',
        'glfw',
        'PyOpenGL',
        'pygltflib',
        'scipy',
        'dill',
        ],
    extras_require=extras,
)
