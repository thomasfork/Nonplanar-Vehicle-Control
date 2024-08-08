'''
glm functions implemented with casadi
meant for joining many together, ie. functions for transforms for wheels of a car
'''

import casadi as ca

def translate(
        M: ca.SX,
        x: ca.SX = 0,
        y: ca.SX = 0,
        z: ca.SX = 0) -> ca.SX:
    ''' translate by x, y, and z values, zero if not provided'''
    M[0,3] += x
    M[1,3] += y
    M[2,3] += z
    return M

def translate_rel(
        M: ca.SX,
        x: ca.SX = 0,
        y: ca.SX = 0,
        z: ca.SX = 0) -> ca.SX:
    ''' translate by x, y, and z values, zero if not provided, but in directions of M'''
    M[0:3,3] += M[:3,:3] @ ca.vertcat(x,y,z)
    return M

def scale(
        M: ca.SX,
        x: ca.SX = 1,
        y: ca.SX = 1,
        z: ca.SX = 1) -> ca.SX:
    ''' scale by x, y, and z values, 1 if not provided '''
    return M @ ca.diag((x,y,z,1))

def rotate(
        M: ca.SX,
        angle: ca.SX,
        x: ca.SX = 1,
        y: ca.SX = 1,
        z: ca.SX = 1) -> ca.SX:
    ''' rotate about vector (x,y,z) by angle (degrees) '''
    th = angle * ca.pi / 180
    c = ca.cos(th)
    s = ca.sin(th)
    norm = ca.norm_2(ca.vertcat(x,y,z))
    x = x / norm
    y = y / norm
    z = z / norm
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    R = ca.vertcat(
        ca.horzcat(cx * x + c, cy * x - z * s, cz * x + y * s, 0),
        ca.horzcat(cx * y + z * s, cy * y + c, cz * y - x * s, 0),
        ca.horzcat(cx * z - y * s, cy * z + x * s, cz * z + c, 0),
        ca.horzcat(0, 0, 0, 1)
    )
    return M @ R
