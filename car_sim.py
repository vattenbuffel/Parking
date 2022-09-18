import numpy as np
from numpy import cos, sin, tan
import pygame
import time
from rolling_average import RollingAverage
from polygon_math import get_area_of_polygon, get_polygon_intersection_points, polygon_from_points, intersection_numba, point_at_t_numba
import numba
from common_functions import GREEN, BLACK, WHITE, BLUE, pos_to_pix
from ParkingArea import ParkingArea
from Wheel import Wheel

"""
Needed functions

env.observation_space.shape[0]
env.action_space.shape[0]
env.reset()
env.render()
obs, reward, done, _ env.step()

"""


#@numba.njit
def model(x, u1, u2, dt, L):
    # x is state vector [x, y, theta, phi]
    # u1 is vel, u2 is steering ang

    theta = x[2, 0]

    v = np.array([
        cos(theta)*u1,
        sin(theta)*u1,
        1/L*tan(u2)*u1,
        0,
    ]).reshape(4, 1)

    x1 = x + v*dt
    x1[3] = u2

    return x1


#@numba.njit
def get_car_corners(x, y, theta):
    h = car_height
    w = car_width
    theta0 = np.arctan2(h/2, w)   #angle at which p1 starts at
    d = ((h/2)**2 + (w)**2)**0.5
    p0 = (h/2*cos(theta + np.pi/2), h/2*sin(theta + np.pi/2))
    p1 = (d * cos(theta + theta0), d * sin(theta + theta0))
    p2 = (d * cos(theta - theta0), d * sin(theta - theta0))
    p3 = (h/2*cos(theta - np.pi/2), h/2*sin(theta - np.pi/2))

    return [p0[0]+x, p1[0]+x, p2[0]+x, p3[0]+x], [p0[1]+y, p1[1]+y, p2[1]+y, p3[1]+y]


def draw_car(car_xs, car_ys):
    p0 = (car_xs[0], car_ys[0])
    p1 = (car_xs[1], car_ys[1])
    p2 = (car_xs[2], car_ys[2])
    p3 = (car_xs[3], car_ys[3])

    p0 = pos_to_pix(*p0, width, height)
    p1 = pos_to_pix(*p1, width, height)
    p2 = pos_to_pix(*p2, width, height)
    p3 = pos_to_pix(*p3, width, height)

    pygame.draw.polygon(screen, BLUE, (p0, p1, p2, p3))

def get_car_parking_area_overlap(pa_xs, pa_ys, car_xs, car_ys):
    xs, ys = get_polygon_intersection_points(pa_xs, pa_ys, car_xs, car_ys)
    if xs != []:
        xs, ys = polygon_from_points(xs, ys)
        intersection_area = get_area_of_polygon(xs, ys)
        car_area = get_area_of_polygon(car_xs, car_ys)
        return intersection_area / car_area

    return 0

def draw_line(x0, y0, x1, y1):
    p0 = pos_to_pix(x0, y0, width, height)
    p1 = pos_to_pix(x1, y1, width, height)
    pygame.draw.line(screen, BLACK, p0, p1)

def draw_lines(xs, ys):
    for i in range(len(xs)-1):
        draw_line(xs[i], ys[i], xs[i+1], ys[i+1])

@numba.njit
def scan_numba(scan_n, car_theta, car_x, car_y, obstacle_xs, obstacle_ys):
    scan_res = {}
    for ang in np.linspace(0, np.deg2rad(360), scan_n):
        x = car_x + np.cos(ang + car_theta)
        y = car_y + np.sin(ang + car_theta)
        l = car_x, car_y, x, y
        for i in range(len(obstacle_xs)-1):
            obstacle_x0, obstacle_y0 = obstacle_xs[i], obstacle_ys[i]
            obstacle_x1, obstacle_y1 = obstacle_xs[i+1], obstacle_ys[i+1]

            t1, t2 = intersection_numba(car_x, car_y, x, y, obstacle_x0, obstacle_y0, obstacle_x1, obstacle_y1)

            if t1 >= 0 and t2 <= 1 and t2 >= 0:
                p = point_at_t_numba(car_x, car_y, x, y, t1)
                d = ((p[0] - car_x)**2 + (p[1] - car_y)**2)**0.5
            else:
                p = None
                d = None

            if ang not in scan_res:
                scan_res[ang] = (d, l, p)
            else:
                if d is not None :
                    if ang in scan_res:
                        d0,_,_ = scan_res[ang]
                        if d0 is None  or d < d0:
                            scan_res[ang] = (d, l, p)
                    else:
                        scan_res[ang] = (d, l, p)

    return  scan_res

def draw_scan_res(scan_res):
    for key in scan_res:
        d, l, p = scan_res[key]
        if p is not None:
            draw_line(l[0], l[1], p[0], p[1])

#@numba.njit
def clean_pa_scan_lines(pa_scan, obstacle_scan):
    for key in pa_scan:
        obs_d, obs_l, obs_p = obstacle_scan[key]
        pa_d, pa_l, pa_p = pa_scan[key]

        if pa_d is None:
            continue

        if pa_d > obs_d:
            pa_scan[key] = None, pa_l, None



# width, height = (1280, 1280)
width, height = (720, 720)
screen = pygame.display.set_mode((width,height))
pygame.font.init()
my_font = pygame.font.SysFont(None, 30)
fps = RollingAverage(100)
t_start = time.time()
draw_scan_lines = False


x = np.zeros((4, 1))
u1 = 0
u2 = 0
dt = 0.5
L = 80
car_width = L
car_height = 50
car_xs , car_ys = get_car_corners(x[0, 0], x[1, 0], x[2, 0])
car_xs.append(car_xs[0]), car_ys.append(car_ys[0]) # Make the polygon closed

wheel_width = 30
wheel_height = 10
wheel1 = Wheel(wheel_width, wheel_height, car_width - wheel_width/2 - 5, -car_height/2 + wheel_height/2 + 5)
wheel2 = Wheel(wheel_width, wheel_height, car_width - wheel_width/2 - 5, car_height/2 - wheel_height/2 - 5)
wheel3 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, car_height/2 - wheel_height/2 - 5)
wheel4 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, -car_height/2 + wheel_height/2 + 5)

pa = ParkingArea([100, 300, 300, 100, 100], [100, 100, 300, 300, 100])

map_xs = [-350, 350, 350, 50, 50, -100, -100, -350, -350]
map_ys  = [-350, -350, 350, 350, 100, 100, 350, 350, -350]


while True:
    t_start = time.time()

    screen.fill(WHITE)
    pa.draw(screen)

    x_ = x[0, 0]
    y = x[1, 0]
    theta = x[2, 0]
    phi = x[3, 0]

    map_scan_res = scan_numba(361, theta, x_, y, map_xs, map_ys)
    pa_scan_res = scan_numba(361, theta, x_, y, pa.xs, pa.ys)
    clean_pa_scan_lines(pa_scan_res, map_scan_res)

    # Draw stuff
    draw_car(car_xs[:-1], car_ys[:-1])
    wheel1.draw(x_, y, theta, phi, screen)
    wheel2.draw(x_, y, theta, phi, screen)
    wheel3.draw(x_, y, theta, 0, screen)
    wheel4.draw(x_, y, theta, 0, screen)
    draw_lines(map_xs, map_ys)
    if draw_scan_lines:
        draw_scan_res(map_scan_res)
        draw_scan_res(pa_scan_res)


    car_pa_overlap = get_car_parking_area_overlap(pa.xs, pa.ys, car_xs, car_ys)
    
    text_surface = my_font.render(f"u1: {u1:.2f}, u2: {np.rad2deg(u2):.2f}, fps: {fps.get():.2f}, pa: {car_pa_overlap:.2f}", True, BLACK)
    screen.blit(text_surface, (50,50))

    pygame.display.update()

    for events in pygame.event.get():
        if events.type == pygame.QUIT:
            import sys
            sys.exit(0)
        elif events.type == pygame.KEYDOWN:
            if events.dict['unicode'] == 'w':
                u1 = 5
            elif events.dict['unicode'] == 'a':
                u2 = np.deg2rad(30)
            elif events.dict['unicode'] == 'd':
                u2 = np.deg2rad(-30)
            elif events.dict['unicode'] == 's':
                u1 = -5
            elif events.dict['unicode'] == '\x1b': # esc
                exit(0)
            elif events.dict['unicode'] == '\x1b': # esc
                exit(0)
            elif events.dict['unicode'] == ' ':
                x = np.zeros((4, 1))
        elif events.type == pygame.KEYUP:
            if events.dict['unicode'] == 'w':
                u1 = 0
            elif events.dict['unicode'] == 'a':
                u2 = 0
            elif events.dict['unicode'] == 'd':
                u2 = 0
            elif events.dict['unicode'] == 's':
                u1 = 0


    # Update car state and check for collision
    x_temp = model(x, u1, u2, dt, L)
    car_xs_temp, car_ys_temp = get_car_corners(x_temp[0, 0], x_temp[1, 0], x_temp[2, 0])
    car_xs_temp.append(car_xs_temp[0]), car_ys_temp.append(car_ys_temp[0]) # Make the polygon closed
    if ([], []) == get_polygon_intersection_points(car_xs_temp, car_ys_temp, map_xs, map_ys):
        x = x_temp
        car_xs, car_ys = car_xs_temp, car_ys_temp
    else:
        oaietnhoian =5

    t_end = time.time()
    fps.update(1/(t_end - t_start))


