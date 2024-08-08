''' polytopic obstacles '''
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import List

import numpy as np

from vehicle_3d.pytypes import PythonMsg, Position, BodyPosition, OrientationQuaternion, \
    NestedPythonMsg

from vehicle_3d.visualization.shaders import vtype

@dataclass
class BaseObstacle(PythonMsg, ABC):
    '''
    Base class for 3D obstacles
    '''

@dataclass
class BasePolytopeObstacle(BaseObstacle):
    '''
    Base class for 3D polytope obstacles

    O = {x : Ab*x <= b}       (Hyperplane form)
    '''

    A: np.ndarray = field(default = None)  # single ended constraints
    b: np.ndarray = field(default = None)

    color: List[float] = field(default = np.array([1,0,0,1]))

    @abstractmethod
    def update(self):
        '''
        Update single-sided hyperplane form of the polytope:

        P = {x : A @ x <= b}
        '''
        return

@dataclass
class RectangleObstacle(BasePolytopeObstacle, NestedPythonMsg):
    '''
    Stores a rectangular prism and computes polytope representations from it.

    Can be used for 2D by ignoring the third axis and third and sixth faces,
    which correspond to the top and bottom faces respectively.

    attributes:
        x: Position of the center of the obstacle
        q: Orientation of the obstacle
        w: width of the obstacle in each dimension

    after changing the above attributes, call update() to
    update the hyperplane representation A @ x <= b
    '''
    x: Position = field(default = None)
    q: OrientationQuaternion = field(default = None)
    w: BodyPosition = field(default = None)

    def __post_init__(self):
        if self.w is None:
            self.w = BodyPosition(x1 = 1, x2 = 1, x3 = 1)
        super().__post_init__()
        self.update()

    def update(self):
        ''' update polytope representation if position, orientation, and width have changed '''

        A1 = self.q.Rinv() / self.w.to_vec()
        b1 = A1 @ self.x.to_vec() + 0.5

        A2 =-A1
        b2 = A2 @ self.x.to_vec() + 0.5

        self.A = np.concatenate([A1, A2])
        self.b = np.concatenate([b1, b2])

    def triangulate_obstacle(self, lines = False):
        ''' generate data for plotting in opengl '''
        e = np.array([[1,1,1,],
                      [1,1,-1],
                      [1,-1,1],
                      [-1,1,1],
                      [1,-1,-1],
                      [-1,1,-1],
                      [-1,-1,1],
                      [-1,-1,-1]]).T

        x = self.q.R() @ e * self.w.to_vec()[:, np.newaxis]/2
        x = x.T + self.x.to_vec()

        Vertices = np.zeros(x.shape[0], dtype=vtype)
        Vertices['a_position'] = x
        Vertices['a_normal'] = np.array([0,0,1])
        Vertices['a_color'] = self.color if not lines else self.color * 0.5

        if not lines:
            I = np.array([0,2,1, 1,2,4, 0,1,3, 1,5,3, 0,3,2, 2,3,6,
                          1,4,5, 5,4,7, 3,5,6, 5,7,6, 2,6,4, 4,6,7], dtype = np.uint32)
        else:
            I = np.array([0,1,0,2,0,3,1,4,1,5,2,4,2,6,3,5,3,6,4,7,5,7,6,7], dtype = np.uint32)

        return Vertices, I
