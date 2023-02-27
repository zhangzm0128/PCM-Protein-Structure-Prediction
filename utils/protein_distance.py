import numpy as np
def distance2(a1, a2):
    x1, y1, z1 = a1
    x2, y2, z2 = a2
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    return dx*dx + dy*dy + dz*dz

def angle(a1, a2, a3):
    x1, y1, z1 = a1
    x2, y2, z2 = a2
    x3, y3, z3 = a3
    v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    v2 = np.array([x2 - x3, y2 - y3, z2 - z3])
    l1 = np.sqrt(np.dot(v1, v1))
    l2 = np.sqrt(np.dot(v2, v2))
    cosa = np.dot(v1, v2) / (l1 * l2)
    return np.degrees(np.arccos(cosa))

def dihedral(a1, a2, a3, a4):
    p = np.array([a1,a2, a3, a4])
    v1 = p[1] - p[0]
    v2 = p[1] - p[2]
    v3 = p[3] - p[2]
    # Take the cross product between v1-v2 and v2-v3
    v1xv2 = _cross(v1, v2)
    v2xv3 = _cross(v2, v3)
    # Now find the angle between these cross-products
    l1 = np.sqrt(np.dot(v1xv2, v1xv2))
    l2 = np.sqrt(np.dot(v2xv3, v2xv3))
    cosa = np.dot(v1xv2, v2xv3) / (l1 * l2)

    #cosa = np.dot(v1xv2, v2xv3) / np.sqrt(np.dot(v1xv2, v1xv2) * np.dot(v2xv3, v2xv3))
    cosa = np.clip(cosa, -1, 1)

    return np.sign(np.dot(v3, v1xv2)) * np.degrees(np.arccos(cosa))

    # if np.dot(v3, v1xv2) <= 0.0:
    #     return np.degrees(np.arccos(cosa))
    # else:
    #     return -np.degrees(np.arccos(cosa))

def _cross(v1, v2):
    """ Computes the cross-product """
    # Can't use np.cross for pypy, since it's not yet implemented
    return np.array([v1[1]*v2[2] - v1[2]*v2[1],
                     v1[2]*v2[0] - v1[0]*v2[2],
                     v1[0]*v2[1] - v1[1]*v2[0]])