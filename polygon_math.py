import matplotlib.pyplot as plt
import numba
import numpy as np
from itertools import permutations

@numba.njit
def point_at_t_numba(x0, y0, x1, y1, t):
    x = x0 + t*(x1 - x0)
    y = y0 + t*(y1 - y0)
    return x, y

@numba.njit
def intersection_numba(xa0, ya0, xa1, ya1, xb0, yb0, xb1, yb1):
    # Get point t1 and t2 where l1 (a points) and l2 (b points) intersect
    xa0 = np.round(xa0, 9)
    ya0 = np.round(ya0, 9)
    xa1 = np.round(xa1, 9)
    ya1 = np.round(ya1, 9)

    xb0 = np.round(xb0, 9)
    yb0 = np.round(yb0, 9)
    xb1 = np.round(xb1, 9)
    yb1 = np.round(yb1, 9)

    dxa = xa1 - xa0
    dya = ya1 - ya0
    dxb = xb1 - xb0
    dyb = yb1 - yb0

    t1, t2 = -1, -1
    div = dxb*dya - dyb*dxa
    if div == 0:
        return t1, t2

    t2 = ((yb0 - ya0)*dxa - (xb0 - xa0)*dya) / div

    if dxa != 0:
        t1 = (xb0 - xa0 + dxb*t2)/dxa
    else:
        # Check if xa0 is on l2 at point given by t2
        x, y = point_at_t_numba(xb0, yb0, xb1, yb1, t2)
        if np.abs(x - xa0) < 1e-10:
            # Calculate where l1 and l2 intersect
            t1 = (y - ya0)/dya

    return t1, t2

#@numba.njit
def cross_2d_numba(x0, y0, x1, y1):
    return (x0 * y1) - (y0 * x1)

#@numba.njit
def which_side_of_line(x0, y0, x1, y1, px, py):
    # Return a negative or positive number based on which side p is of l
    l1x = x1 - x0
    l1y = y1 - y0

    l2x = px - x0
    l2y = py - y0
    return cross_2d_numba(l1x, l1y, l2x, l2y)


#@numba.njit
def point_in_polygon(xs, ys, px, py):
    assert xs[0] == xs[-1]
    assert ys[0] == ys[-1]
    # Return true if p is in polygon defined xs and ys
    sign = np.sign(which_side_of_line(xs[0], ys[0], xs[1], ys[1], px, py))
    for i in range(len(xs)-1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i+1], ys[i+1]
        if sign != np.sign(which_side_of_line(x0, y0, x1, y1, px, py)):
            return False

    return True

#@numba.njit
def get_points_in_polygon(xs, ys, pxs, pys):
    assert xs[0] == xs[-1]
    assert ys[0] == ys[-1]
    # Get all points that are inside polygon
    # Polygon is defined by the lines defined by xs and ys
    # Points are defined by pxs and pys
    intersection_xs = []
    intersection_ys = []

    for i in range(len(pxs)):
        px, py = pxs[i], pys[i]

        if point_in_polygon(xs, ys, px, py):
            intersection_xs.append(px)
            intersection_ys.append(py)
    
    return intersection_xs, intersection_ys

#@numba.njit
def get_intersections_between_lines(xsa, ysa, xsb, ysb, remove_duplicates=True):
    # line0 = (xsa[0], ysa[0], xsa[1], ysa[1])
    # line1 = (xsa[1], ysa[1], xsa[2], ysa[2])
    # etc...
    # Return all points where lines defined by a and b points intersect

    intersection_points = []


    for i in range(len(xsa)-1):
        x0a, y0a = xsa[i], ysa[i]
        x1a, y1a = xsa[i+1], ysa[i+1]

        for j in range(len(xsb)-1):
            x0b, y0b = xsb[j], ysb[j]
            x1b, y1b = xsb[j+1], ysb[j+1]

            t1, t2 = intersection_numba(x0a, y0a, x1a, y1a, x0b, y0b, x1b, y1b)

            if t1 <= 1 and t1 >= 0 and t2 <= 1 and t2 >= 0:
                p = point_at_t_numba(x0a, y0a, x1a, y1a, t1)
                if remove_duplicates and p in intersection_points:
                    continue
                intersection_points.append(p)

    xs, ys = [], []
    for p in intersection_points:
        xs.append(p[0])
        ys.append(p[1])

    return xs, ys

#@numba.njit
def get_polygon_intersection_points(xsa, ysa, xsb, ysb):
    # Given 2 polygons return all points that make up the intersetion
    assert xsa[0] == xsa[-1]
    assert ysa[0] == ysa[-1]
    assert xsb[0] == xsb[-1]
    assert ysb[0] == ysb[-1]
    xs, ys = get_intersections_between_lines(xsa, ysa, xsb, ysb)

    xs_, ys_ = get_points_in_polygon(xsa, ysa, xsb[:-1], ysb[:-1])
    xs.extend(xs_), ys.extend(ys_)

    xs_, ys_ = get_points_in_polygon(xsb, ysb, xsa[:-1], ysa[:-1])
    xs.extend(xs_), ys.extend(ys_)

    return xs, ys


#@numba.njit
def is_close(val1, val2):
    return np.abs(val1 - val2) < 1e-10


#@numba.njit
def polygon_is_convex(xs, ys):
    # See if polygon given by xs and ys is convex by checking if any of the lines overlap
    assert xs[0] == xs[-1]
    assert ys[0] == ys[-1]

    for i in range(len(xs)-1):
        x0a, y0a = xs[i], ys[i]
        x1a, y1a = xs[i+1], ys[i+1]

        for j in range(len(xs)-1):
            if i==j:
                continue

            x0b, y0b = xs[j], ys[j]
            x1b, y1b = xs[j+1], ys[j+1]

            t1, t2 = intersection_numba(x0a, y0a, x1a, y1a, x0b, y0b, x1b, y1b)

            # The lines are allowd to touch on the endpoints
            if is_close(t1, 1) or is_close(t1, 0):
               continue
            if is_close(t2, 1) or is_close(t2, 0):
                continue

            if(t1 > 0 and t1 < 1):
                return False
            if(t2 > 0 and t2 < 1):
                return False

    return True


#@numba.njit
def perm_polygon_is_convex(xs, ys, perm):
    xs_, ys_ = [], []
    for i in perm:
        xs_.append(xs[i])
        ys_.append(ys[i])
    if polygon_is_convex(xs_, ys_):
        return True

    return False
    
            

#@numba.njit
def close_permutations(perms):
    perms_closed = []
    for p in perms:
        p = list(p)
        p.append(p[0])
        perms_closed.append(p)
    
    return perms_closed

def polygon_from_points(xs, ys):
    # Return a simple polygon consiting of points given by xs and ys

    # Create all possible combinations of verticies
    perms = set(permutations(range(len(xs))))
    perms_closed = close_permutations(perms)

    for perm in perms_closed:
        if perm_polygon_is_convex(xs, ys, perm):
            xs_, ys_ = [], []
            for i in perm:
                xs_.append(xs[i])
                ys_.append(ys[i])
            return  xs_, ys_
    
    assert False, "There must be a convex simple polygon"

#@numba.njit
def determinant_2d(a, b, c, d):
    return a*d - b*c

#@numba.njit
def get_area_of_polygon(xs, ys):
    assert xs[0] == xs[-1]
    assert ys[0] == ys[-1]

    A = 0
    for i in range(len(xs)-1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i+1], ys[i+1]

        A += determinant_2d(x0, x1, y0, y1)
    
    if A < 0:
        A = -A

    return A/2

# Test that point on different sides of line give different signs
sign1 = which_side_of_line(2, 2, 4, 2, 3, 3)
sign2 = which_side_of_line(4, 2, 2, 2, 3, 3)
assert np.sign(sign1) != np.sign(sign2)

# Test that points inside a polygon have the same sign
sign1 = which_side_of_line(2, 2, 4, 2, 3, 3)
sign2 = which_side_of_line(4, 2, 4, 4, 3, 3)
sign3 = which_side_of_line(4, 4, 2, 4, 3, 3)
sign4 = which_side_of_line(2, 4, 2, 2, 3, 3)
assert np.sign(sign1) == np.sign(sign2) == np.sign(sign3) == np.sign(sign4)

# Test if a point is inside a polygon or not
assert True == point_in_polygon([2, 4, 4, 2, 2], [2, 2, 4, 4, 2], 3, 3)
assert False == point_in_polygon([2, 4, 4, 2, 2], [2, 2, 4, 4, 2], -100, -100)


# Test line intersection at special case where l1 is perpendicular against x-axis, ie x=0 for all y
assert (1, 0) == intersection_numba(1, 1, 1, 0, 1, 0, 3, 2)
assert (1, 0) == intersection_numba(0, 1, 0, 0, 0, 0, 2, 2)
assert (-1, -1) == intersection_numba(0, 0, 1, 0, 1, 1, 0, 1)
xs = [0, 2, 2, 0, 0]
ys = [0, 0, 1, 1, 0]
assert 3 == len(get_intersections_between_lines(xs, ys, [0, 2], [0, 2], remove_duplicates=False)[0])
assert 2 == len(get_intersections_between_lines(xs, ys, [0, 2], [0, 2], remove_duplicates=True)[0])

# Test if points are within a polygon
xs = [0,3,3,0,0]
ys = [0,0,3,3,0]
pxs = [2, 4, 4, 2]
pys = [1, 1, 2, 2]
assert ([2,2], [1,2]) == get_points_in_polygon(xs, ys, pxs, pys)
xs = [100,103,103,100,100]
ys = [0,0,3,3,0]
pxs = [2, 4, 4, 2]
pys = [1, 1, 2, 2]
assert ([], []) == get_points_in_polygon(xs, ys, pxs, pys)


# Find the points that make up the intersection in scenario 1 
xsa = [-2, -1, -1, -2, -2]
ysa = [-2, -2, -1, -1, -2]
xsb = [1, 2, 2, 1, 1]
ysb = [1, 1, 2, 2, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
ps = {(xs[i], ys[i]) for i in range(len(xs))}
assert set() == ps

# Find the points that make up the intersection in scenario 2
xsa = [0, 3, 3, 0, 0]
ysa = [0, 0, 3, 3, 0]
xsb = [1, 2, 2, 1, 1]
ysb = [1, 1, 2, 2, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
ps = {(xs[i], ys[i]) for i in range(len(xs))}
assert {(1,1), (2, 1), (2, 2), (1, 2)} == ps

# Find the points that make up the intersection in scenario 3
xsa = [0, 3, 3, 0, 0]
ysa = [0, 0, 3, 3, 0]
xsb = [2, 4, 4, 2, 2]
ysb = [1, 1, 2, 2, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
ps = set((xs[i], ys[i]) for i in range(len(xs)))
assert {(2, 1), (3, 1), (3, 2), (2, 2)} == ps

# Find the points that make up the intersection in scenario 4
xsa = [1, 2, 2, 1, 1]
ysa = [0, 0, 2, 2, 0]
xsb = [0, 3, 3, 0, 0]
ysb = [1, 1, 3, 3, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
ps = set((xs[i], ys[i]) for i in range(len(xs)))
assert {(1, 1), (2, 1), (2, 2), (1, 2)} == ps


# Find the points that make up the intersection in scenario 4 v2
xsa = [1, 7, 7, 1, 1]
ysa = [0, 0, 3, 3, 0]
xsb = [0, 6, 5, 1, 0]
ysb = [1, 4, 5, 2, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
ps = set((xs[i], ys[i]) for i in range(len(xs)))
assert 4 == len(ps)


# Test if polygon is convex
assert True == polygon_is_convex([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
assert False == polygon_is_convex([0, 1, 0, 1, 0], [0, 1, 1, 0, 0])

# Test creating intersecting polygon from scenario 2
xsa = [0, 3, 3, 0, 0]
ysa = [0, 0, 3, 3, 0]
xsb = [1, 2, 2, 1, 1]
ysb = [1, 1, 2, 2, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
xs, ys = polygon_from_points(xs, ys)
ps = {(xs[i], ys[i]) for i in range(len(xs))}
assert {(1,1), (2, 1), (2, 2), (1, 2)} == ps

# Test creating intersecting polygon from scenario 3
xsa = [0, 3, 3, 0, 0]
ysa = [0, 0, 3, 3, 0]
xsb = [2, 4, 4, 2, 2]
ysb = [1, 1, 2, 2, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
xs, ys = polygon_from_points(xs, ys)
ps = {(xs[i], ys[i]) for i in range(len(xs))}
assert {(2,1), (3,1), (3,2), (2,2)} == ps

# Test creating intersecting polygon from scenario 4
xsa = [1, 2, 2, 1, 1]
ysa = [0, 0, 2, 2, 0]
xsb = [0, 3, 3, 0, 0]
ysb = [1, 1, 3, 3, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
xs, ys = polygon_from_points(xs, ys)
ps = set((xs[i], ys[i]) for i in range(len(xs)))
assert {(1, 1), (2, 1), (2, 2), (1, 2)} == ps

# Test creating intersecting polygon from scenario 4 v2
xsa = [1, 7, 7, 1, 1]
ysa = [0, 0, 3, 3, 0]
xsb = [0, 6, 5, 1, 0]
ysb = [1, 4, 5, 2, 1]
xs, ys = get_polygon_intersection_points(xsa, ysa, xsb, ysb)
xs, ys = polygon_from_points(xs, ys)
ps = set((xs[i], ys[i]) for i in range(len(xs)))
assert 4 == len(ps)

# Test calculating area of polygon
xs = [0, 1, 1, 0, 0]
ys = [0, 0, 1, 1, 0]
assert 1 == get_area_of_polygon(xs, ys)
alpha = np.deg2rad(00)
xs = [0, np.cos(alpha), 2**0.5*np.cos(alpha + np.pi/4), np.cos(alpha + np.pi/2), 0]
ys = [0, np.sin(alpha), 2**0.5*np.sin(alpha + np.pi/4), np.sin(alpha + np.pi/2), 0]
assert 1 == get_area_of_polygon(np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32))

if __name__ == '__main__':
    # Plot scenario 2
    Px = [0, 3, 3, 0, 0]
    Py = [0, 0, 3, 3, 0]

    Qx = [1, 2, 2, 1, 1]
    Qy = [1, 1, 2, 2, 1]

    xs, ys = get_polygon_intersection_points(Px, Py, Qx, Qy)
    xs, ys = polygon_from_points(xs, ys)

    plt.figure(figsize=(8, 8))
    plt.title("Scenario 2")
    plt.axis('equal')
    plt.fill(Px, Py, alpha=0.5, label="P")
    plt.fill(Qx, Qy, alpha=0.5, label="Q")
    plt.fill(xs, ys, alpha=0.5, label="P and Q")
    plt.legend()
    plt.show(block=False)


    # Plot scenario 3
    Px = [0,3,3,0,0]
    Py = [0,0,3,3,0]

    Qx = [2, 4, 4, 2, 2]
    Qy = [1,1,2,2,1]

    xs, ys = get_polygon_intersection_points(Px, Py, Qx, Qy)
    xs, ys = polygon_from_points(xs, ys)

    plt.figure(figsize=(8, 8))
    plt.title("Scenario 3")
    plt.axis('equal')
    plt.fill(Px, Py, alpha=0.5, label="P")
    plt.fill(Qx, Qy, alpha=0.5, label="Q")
    plt.fill(xs, ys, alpha=0.5, label="P and Q")
    plt.legend()
    plt.show(block=False)

    # Plot scenario 4
    Px = [1, 2, 2, 1, 1]
    Py = [0, 0, 2, 2, 0]

    Qx = [0, 3, 3, 0, 0]
    Qy = [1, 1, 3, 3, 1]

    xs, ys = get_polygon_intersection_points(Px, Py, Qx, Qy)
    xs, ys = polygon_from_points(xs, ys)

    plt.figure(figsize=(8, 8))
    plt.title("Scenario 4")
    plt.axis('equal')
    plt.fill(Px, Py, alpha=0.5, label="P")
    plt.fill(Qx, Qy, alpha=0.5, label="Q")
    plt.fill(xs, ys, alpha=0.5, label="P and Q")
    plt.legend()
    plt.show(block=False)

    # Plot scenario 4 v2
    Px = [1, 7, 7, 1, 1]
    Py = [0, 0, 3, 3, 0]

    Qx = [0, 6, 5, 1, 0]
    Qy = [1, 4, 5, 2, 1]

    xs, ys = get_polygon_intersection_points(Px, Py, Qx, Qy)
    xs, ys = polygon_from_points(xs, ys)

    plt.figure(figsize=(8, 8))
    plt.title("Scenario 4 v2")
    plt.axis('equal')
    plt.fill(Px, Py, alpha=0.5, label="P")
    plt.fill(Qx, Qy, alpha=0.5, label="Q")
    plt.fill(xs, ys, alpha=0.5, label="P and Q")
    plt.legend()
    plt.show()
