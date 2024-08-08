'''
basic glm functions

code adapted from glumpy library, https://github.com/glumpy/glumpy/blob/master/glumpy/glm.py

-----------------------------------------------------------------------------
Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
Distributed under the (new) BSD License.
-----------------------------------------------------------------------------

'''
import numpy as np

def translate(
        M: np.ndarray,
        x: float = 0,
        y: float = 0,
        z: float = 0):
    ''' translate by x, y, and z values, zero if not provided'''
    MT = M.copy()
    MT[:3,3] += np.array([x,y,z])
    return MT

def scale(
        M: np.ndarray,
        x: float = 1,
        y: float = 1,
        z: float = 1):
    ''' scale by x, y, and z values, 1 if not provided '''
    return M @ np.diag((x,y,z,1)).astype(M.dtype)

def rotate(
        M: np.ndarray,
        angle: float,
        x: float = 1,
        y: float = 1,
        z: float = 1):
    ''' rotate about vector (x,y,z) by angle (degrees) '''
    angle = np.pi * angle / 180
    c = np.cos(angle)
    s = np.sin(angle)
    n = np.sqrt(x * x + y * y + z * z)
    x = x / n
    y = y / n
    z = z / n
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    R = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, 0],
                  [cx * y + z * s, cy * y + c, cz * y - x * s, 0],
                  [cx * z - y * s, cy * z + x * s, cz * z + c, 0],
                  [0, 0, 0, 1]])
    return M @ R.astype(M.dtype)


def ortho(left, right, bottom, top, znear, zfar):
    """Create orthographic projection matrix
    Parameters
    ----------
    left : float
        Left coordinate of the field of view.
    right : float
        Right coordinate of the field of view.
    bottom : float
        Bottom coordinate of the field of view.
    top : float
        Top coordinate of the field of view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.
    Returns
    -------
    M : array
        Orthographic projection matrix (4x4).
    """

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 / (right - left)
    M[3, 0] = -(right + left) / float(right - left)
    M[1, 1] = +2.0 / (top - bottom)
    M[3, 1] = -(top + bottom) / float(top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 3] = 1.0
    return M


def frustum(left, right, bottom, top, znear, zfar):
    """Create view frustum
    Parameters
    ----------
    left : float
        Left coordinate of the field of view.
    right : float
        Right coordinate of the field of view.
    bottom : float
        Bottom coordinate of the field of view.
    top : float
        Top coordinate of the field of view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.
    Returns
    -------
    M : array
        View frustum matrix (4x4).
    """

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M


def perspective(fovy, aspect, znear, zfar):
    """Create perspective projection matrix
    Parameters
    ----------
    fovy : float
        The field of view along the y axis.
    aspect : float
        Aspect ratio of the view.
    znear : float
        Near coordinate of the field of view.
    zfar : float
        Far coordinate of the field of view.
    Returns
    -------
    M : array
        Perspective projection matrix (4x4).
    """
    h = np.tan(fovy / 360.0 * np.pi) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)
