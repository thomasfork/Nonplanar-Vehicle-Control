# Nonplanar Vehicle Control
This package provides tools for the simulation, modeling, planning and control of ground vehicles.

# Citing
For most intents and purposes, citing my dissertation will suffice. It is avaible open access at https://www.proquest.com/openview/354e67da3819f973e7e9d756c0006a67
```
@phdthesis{fork2024non,
  title={Non-Euclidean Vehicle Motion Models},
  author={Fork, Thomas David},
  year={2024},
  school={University of California, Berkeley}
}
```

Those looking to cite earlier papers may find the following useful:


Original theory and development. Application to a simplified lane-keeping scenario. https://arxiv.org/abs/2104.08427
```
@inproceedings{fork2021models,
      title={Models and Predictive Control for Nonplanar Vehicle Navigation}, 
      author={Thomas Fork and H. Eric Tseng and Francesco Borrelli},
      year={2021},
      booktitle = {2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
      pages = {749-754}
}
```
Extension of theory to offset vehicle reference from surface, application to racelines. https://arxiv.org/abs/2204.09720
```
@inproceedings{fork2022models,
      title={Vehicle Models and Optimal Control on a Nonplanar Surface}, 
      author={Thomas Fork and H. Eric Tseng and Francesco Borrelli},
      year={2022},
      booktitle = {15th International Symposium on Advanced Vehicle Control},
      pages = {749-754}
}
```
Extension of road model to motorcycles, application to racelines. https://arxiv.org/abs/2406.01726
```
@article{fork2024general,
  title={A General 3D Road Model for Motorcycle Racing},
  author={Fork, Thomas and Borrelli, Francesco},
  journal={arXiv preprint arXiv:2406.01726},
  year={2024}
}
```

# License
MIT License Copyright (c) 2023 Thomas Fork


# Setup and Usage
This code has been developed and tested on Windows 10 and 11, as well as Ubuntu 20.04 and 22.04. It has also functioned on newer Mac operating systems, but 3D graphics support is not guaranteed.

Intructions are provided below for Linux, similar steps may work on Windows and Mac.

## Download the code and install project python package
```
git clone https://github.com/thomasfork/Nonplanar-Vehicle-Control.git
cd src
pip3 install -e .
```

## Running the code
Self-contained code examples are provided in the ```examples/``` folder of this repository.

## Known Issues
Nonlinear optimization problems solved with IPOPT using CasADi 3.6+ default to a newer version of MUMPS which is known to perform worse on certain problems than earlier versions, particularly those using orthogonal collocation. Downgrading to CasADi version 3.5.5 is strongly recommended. On Linux it is possible to use HSL solvers, using https://github.com/coin-or-tools/ThirdParty-HSL. All examples will default to HSL solvers if they are installed.


# Acknowledgements


## 3D Graphics
The 3D graphics code is largely based on examples and tutorials from the following:

Glumpy 
https://glumpy.github.io/index.html

Learn OpenGL
https://learnopengl.com/
The learnopengl.com tutorials were written by Joe de Vries, 
more information can be found here: https://learnopengl.com/About
or https://twitter.com/JoeyDeVriez


## Computational libraries
None of this work would have been possible without powerful symbolic algebra and optimization
tools, which casadi has done a stellar job of providing. More information on the contributors
can be found on their webpage: https://web.casadi.org/publications/
and their github: https://github.com/casadi/casadi


## 3D Assets:
The skybox texure used in the 3D graphics was downloaded from learnopengl.com and is CC BY 4.0
https://creativecommons.org/licenses/by-nc/4.0/

The body and stationary wheels for the 3D racecar asset was made by Sketchfab user Lexyc16,
is licensed CC BY-NC 4.0, and can be downloaded from https://skfb.ly/opvyH

The 3D asset for the individual racecar wheels (the ones that spin) was made by Sketchfab user SDC Performance, 
is licensed CC BY 4.0, and can be downloaded from https://skfb.ly/oovNn

The 3D motorcycle asset was made by Sketchfab user nouter2077,
is licensed CC BY 4.0, and can be downloaded from https://skfb.ly/oIYFv
