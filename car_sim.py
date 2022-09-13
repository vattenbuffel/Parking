import numpy as np
from numpy import cos, sin, tan
import pygame
import time
from rolling_average import RollingAverage
from polygon_math import get_area_of_polygon, get_polygon_intersection_points, polygon_from_points
import numba

class ParkingArea:
    def __init__(self, xs, ys) -> None:
        self.xs = xs
        self.ys = ys

    def draw(self):
        ps = tuple((pos_to_pix(self.xs[i], self.ys[i])) for i in range(len(self.xs)) )
        pygame.draw.polygon(screen, GREEN, ps)

class Wheel:
    def __init__(self, w, h, x_offset_to_car, y_offset_to_car) -> None:
        self.alpha = np.arctan2(h/2, w/2)
        self.r = ((w/2)**2 + (h/2)**2)**0.5
        self.x_offset_to_car = x_offset_to_car
        self.y_offset_to_car = y_offset_to_car
        self.d_offset_to_car = (x_offset_to_car**2 + y_offset_to_car**2)**0.5
        self.alpha_offset_to_car = np.arctan2(y_offset_to_car, x_offset_to_car)

    def draw(self, car_x, car_y, theta, phi):
        r = self.r
        alpha = self.alpha

        p0 = (r * np.cos(theta + alpha + phi), r * np.sin(theta + alpha + phi))
        p1 = (r * np.cos(theta + np.pi - alpha + phi), r * np.sin(theta + np.pi - alpha + phi))
        p2 = (r * np.cos(theta + alpha + np.pi + phi), r * np.sin(theta + alpha + np.pi + phi))
        p3 = (r * np.cos(theta - alpha + phi), r * np.sin(theta - alpha + phi))

        d = self.d_offset_to_car
        alpha = self.alpha_offset_to_car
        x_offset = d * np.cos(alpha + theta)  
        y_offset = d * np.sin(alpha + theta)  

        p0 = add_xy_and_offset(p0, car_x+x_offset, car_y+y_offset)
        p1 = add_xy_and_offset(p1, car_x+x_offset, car_y+y_offset)
        p2 = add_xy_and_offset(p2, car_x+x_offset, car_y+y_offset)
        p3 = add_xy_and_offset(p3, car_x+x_offset, car_y+y_offset)

        pygame.draw.polygon(screen, BLACK, (p0, p1, p2, p3))

# @numba.njit
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


def pos_to_pix(x, y):
    x += width/2
    y = height - y - 1
    y -= height/2
    return x, y

def add_xy_and_offset(p, x, y):
    return p[0] + x + width/2, height/2 - y - 1 - p[1]

#@numba.njit
def get_car_xs_ys(x, y, theta):
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

    p0 = pos_to_pix(*p0)
    p1 = pos_to_pix(*p1)
    p2 = pos_to_pix(*p2)
    p3 = pos_to_pix(*p3)

    pygame.draw.polygon(screen, BLUE, (p0, p1, p2, p3))

def get_car_parking_area_overlap(pa_xs, pa_ys, car_xs, car_ys):
    car_xs.append(car_xs[0])
    car_ys.append(car_ys[0])
    xs, ys = get_polygon_intersection_points(pa_xs, pa_ys, car_xs, car_ys)
    if xs != []:
        xs, ys = polygon_from_points(xs, ys)
        intersection_area = get_area_of_polygon(xs, ys)
        car_area = get_area_of_polygon(car_xs, car_ys)
        return intersection_area / car_area

    return 0



x = np.zeros((4, 1))
u1 = 0
u2 = 0
dt = 0.001
L = 80

# width, height = (1280, 1280)
width, height = (720, 720)
arrow_size = 16
car_width = L
car_height = 50
wheel_width = 30
wheel_height = 10
screen = pygame.display.set_mode((width,height))
pygame.font.init()
my_font = pygame.font.SysFont(None, 30)
fps = RollingAverage(100)
t_start = time.time()

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255, 0 ,0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
size = (300, 300)

wheel1 = Wheel(wheel_width, wheel_height, car_width - wheel_width/2 - 5, -car_height/2 + wheel_height/2 + 5)
wheel2 = Wheel(wheel_width, wheel_height, car_width - wheel_width/2 - 5, car_height/2 - wheel_height/2 - 5)
wheel3 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, car_height/2 - wheel_height/2 - 5)
wheel4 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, -car_height/2 + wheel_height/2 + 5)

pa = ParkingArea([100, 300, 300, 100, 100], [100, 100, 300, 300, 100])


while True:
    t_start = time.time()

    screen.fill(WHITE)
    pa.draw()

    x_ = x[0, 0]
    y = x[1, 0]
    theta = x[2, 0]
    phi = x[3, 0]

    car_xs , car_ys = get_car_xs_ys(x_, y, theta)
    draw_car(car_xs.copy(), car_ys.copy())
    # Draw wheels
    wheel1.draw(x_, y, theta, phi)
    wheel2.draw(x_, y, theta, phi)
    wheel3.draw(x_, y, theta, 0)
    wheel4.draw(x_, y, theta, 0)

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



    for i in range(100):
        x = model(x, u1, u2, dt, L)

    t_end = time.time()
    fps.update(1/(t_end - t_start))



