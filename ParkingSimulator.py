import numpy as np
from numpy import cos, sin, tan
import pygame
import time
from rolling_average import RollingAverage
from polygon_math import get_area_of_polygon, get_polygon_intersection_points, point_in_polygon, polygon_from_points, intersection_numba, point_at_t_numba
import numba
from common_functions import GREEN, BLACK, WHITE, BLUE, pos_to_pix
from ParkingArea import ParkingArea
from Wheel import Wheel


class ParkingSimulator:
    def __init__(self) -> None:
        self.width, self.height = (720, 720)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.font.init()
        self.my_font = pygame.font.SysFont(None, 30)
        self.fps = RollingAverage(100)
        self.t_start = time.time()
        self.draw_scan_lines = False


        self.x = np.zeros([4, 1])
        self.u1 = 0
        self.u2 = 0
        self.dt = 0.5
        self.L = 80
        self.car_width = self.L
        self.car_height = 50
        self.car_xs , self.car_ys = get_car_corners(self.x[0, 0], self.x[1, 0], self.x[2, 0], self.car_width, self.car_height)
        self.car_xs.append(self.car_xs[0]), self.car_ys.append(self.car_ys[0]) # Make the polygon closed
        self.reward = -1

        wheel_width = 30
        wheel_height = 10
        self.wheel1 = Wheel(wheel_width, wheel_height, self.car_width - wheel_width/2 - 5, -self.car_height/2 + wheel_height/2 + 5)
        self.wheel2 = Wheel(wheel_width, wheel_height, self.car_width - wheel_width/2 - 5, self.car_height/2 - wheel_height/2 - 5)
        self.wheel3 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, self.car_height/2 - wheel_height/2 - 5)
        self.wheel4 = Wheel(wheel_width, wheel_height, wheel_width/2 + 5, -self.car_height/2 + wheel_height/2 + 5)

        self.pa = ParkingArea([100, 300, 300, 100, 100], [100, 100, 300, 300, 100])
        self.car_pa_overlap = -1
        self.map_xs = [-350, 350, 350, 50, 50, -100, -100, -350, -350]
        self.map_ys  = [-350, -350, 350, 350, 100, 100, 350, 350, -350]

        self.reset()

        # I don't really know what this is but it's needed for compbability with gym
        class space:
            def __init__(self, shape_size) -> None:
                self.shape = [shape_size]
        self.observation_space = space(1) # Change 361 to be a param
        # self.observation_space = space(361*2+3) # Change 361 to be a param
        
        self.action_space = space(2)


    def reset(self):
        # TODO Make it spawn at random locations. To do this restructure obstacles and map to be convex
        self.x = np.zeros((4, 1))

        # Do this better...
        # self.x[0,0] = np.random.randint(-100, 100)
        # self.x[1,0] = np.random.randint(-100, 100)
        # self.x[2,0] = np.random.randint(0, 360)
        # while True:
        #     self.x[0,0] = np.random.randint(-self.width/2, self.width/2)
        #     self.x[1,0] = np.random.randint(-self.height/2, self.height/2)
        #     self.x[2,0] = np.random.randint(0, 360)

        #     xs, ys = get_car_corners(self.x[0,0], self.x[1, 0], self.x[3, 0], self.car_width, self.car_height)

        #     if valid_position(xs, ys, self.map_xs, self.map_ys):
        #         break


        self.map_scan_res = scan_numba(361, self.x[2, 0], self.x[0, 0], self.x[1, 0], self.map_xs, self.map_ys)
        self.pa_scan_res = scan_numba(361, self.x[2, 0], self.x[0, 0], self.x[1, 0], self.pa.xs, self.pa.ys)
        clean_pa_scan_lines(self.pa_scan_res, self.map_scan_res)

        self.obs = create_obs(self.map_scan_res, self.pa_scan_res, self.x)
        return self.obs


    def step(self, u1u2):
        # u1u2 is a vector of u1 and u2. u1 and u2 are values between -1 and 1 giving the percentage value between min and max
        self.t_start = time.time()
        if isinstance(u1u2, np.ndarray):
            u1u2 = u1u2.reshape(-1)

        self.u1, self.u2 = u1u2[0]*5, u1u2[1]*np.deg2rad(30) # Change 5 and 30 to be params

        # Update car state and check for collision
        x_temp = model(self.x, self.u1, self.u2, self.dt, self.L)
        car_xs_temp, car_ys_temp = get_car_corners(x_temp[0, 0], x_temp[1, 0], x_temp[2, 0], self.car_width, self.car_height)
        car_xs_temp.append(car_xs_temp[0]), car_ys_temp.append(car_ys_temp[0]) # Make the polygon closed
        if ([], []) == get_polygon_intersection_points(car_xs_temp, car_ys_temp, self.map_xs, self.map_ys):
            self.x = x_temp
            self.car_xs, self.car_ys = car_xs_temp, car_ys_temp

        self.car_pa_overlap = get_car_parking_area_overlap(self.pa.xs, self.pa.ys, self.car_xs, self.car_ys)

        x_ = self.x[0, 0]
        y = self.x[1, 0]
        theta = self.x[2, 0]

        # self.map_scan_res = scan_numba(361, theta, x_, y, self.map_xs, self.map_ys)# Change 361 to be a parm
        # self.pa_scan_res = scan_numba(361, theta, x_, y, self.pa.xs, self.pa.ys)
        # clean_pa_scan_lines(self.pa_scan_res, self.map_scan_res)

        obs = create_obs(self.map_scan_res, self.pa_scan_res, self.x)
        self.reward = calculate_reward(self.pa_scan_res, self.x, self.pa.xs, self.pa.ys, self.car_pa_overlap)
        done = False
        unknown = None

        return obs, self.reward, done, unknown

    def render(self, consume_events=True):
        # consume events is needed to be done when training
        self.screen.fill(WHITE)
        self.pa.draw(self.screen)

        x_ = self.x[0, 0]
        y = self.x[1, 0]
        theta = self.x[2, 0]
        phi = self.x[3, 0]


        # Draw stuff
        draw_car(self.car_xs[:-1], self.car_ys[:-1], self.screen, self.width, self.height)
        self.wheel1.draw(x_, y, theta, phi, self.screen)
        self.wheel2.draw(x_, y, theta, phi, self.screen)
        self.wheel3.draw(x_, y, theta, 0, self.screen)
        self.wheel4.draw(x_, y, theta, 0, self.screen)
        draw_lines(self.map_xs, self.map_ys, self.screen, self.width, self.height)
        if self.draw_scan_lines:
            draw_scan_res(self.map_scan_res, self.screen, self.width, self.height)
            draw_scan_res(self.pa_scan_res, self.screen, self.width, self.height)

        t_end = time.time()
        self.fps.update(1/(t_end - self.t_start))
        str_ = f"u1: {self.u1:.2f}, u2: {np.rad2deg(self.u2):.2f}, pa: {self.car_pa_overlap:.2f}, fps: {self.fps.get():.2f}, reward: {self.reward:.2f}"
        text_surface = self.my_font.render(str_, True, BLACK)
        self.screen.blit(text_surface, (50,50))

        pygame.display.update()

        if consume_events:
            for _ in pygame.event.get():
                pass

@numba.njit
def valid_position(car_xs, car_ys, map_xs, map_ys):
    # this doesn't work as point_in_polygon needs the polygon to be convex and map is not
    for i in range(len(car_xs)):
        if not point_in_polygon(map_xs, map_ys, car_xs[i], car_ys[i]):
            return False

    return True

# @numba.njit
def create_obs(obs_scan, pa_scan, x):
    # obs = np.zeros(len(obs_scan) + len(pa_scan) + 3)
    # i = 0
    # for key in obs_scan:
    #     d, l, p = obs_scan[key]
    #     obs[i] = d
    #     i += 1
    # for key in pa_scan:
    #     d, l, p = obs_scan[key]
    #     obs[i] = d
    #     i += 1

    # obs[i] = x[0,0]
    # obs[i+1] = x[1,0]
    # obs[i+2] = x[2,0]

    # return obs

    return x[0]
    


@numba.njit
def calculate_reward(pa_scan, x, pa_xs, pa_ys, pa_overlap):
    # weight_scan_line_touching_pa = 0
    # weight_pa_overlap = 0
    # weight_distance_to_pa = 1000
    # reward_distance_to_pa_max = 150

    # reward_pa_scan = 0
    # for key in pa_scan:
    #     d, l, p = pa_scan[key]
    #     if d == None:
    #         continue
    #     reward_pa_scan += weight_scan_line_touching_pa

    # reward_pa_overlap = pa_overlap * weight_pa_overlap
    
    # d = np.maximum(1e-5, ((pa_xs[0] - x[0,0])**2 + (pa_ys[0] - x[1,0])**2)**0.5) # Calculate distance to first pa xs and ys point. Not the "true" distance, whatever that is, but good enough to encourage it to move closer, I think
    # reward_distance_to_pa = np.minimum(reward_distance_to_pa_max, 1/d*weight_distance_to_pa)

    # return reward_pa_scan + reward_pa_overlap + reward_distance_to_pa
    return x[0, 0]

@numba.njit
def clip(val, min_, max_):
    return np.minimum(max_, np.maximum(min_, val))

@numba.njit
def model(x, u1, u2, dt, L):
    # x is state vector [x, y, theta, phi]
    # u1 is vel, u2 is steering ang
    u1 = clip(u1, -5, 5) # Change 5 to be param
    u2 = clip(u2, -np.deg2rad(30), np.deg2rad(30)) # Change 30 to be param

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


@numba.njit
def get_car_corners(x, y, theta, car_width, car_height):
    h = car_height
    w = car_width
    theta0 = np.arctan2(h/2, w)   #angle at which p1 starts at
    d = ((h/2)**2 + (w)**2)**0.5
    p0 = (h/2*cos(theta + np.pi/2), h/2*sin(theta + np.pi/2))
    p1 = (d * cos(theta + theta0), d * sin(theta + theta0))
    p2 = (d * cos(theta - theta0), d * sin(theta - theta0))
    p3 = (h/2*cos(theta - np.pi/2), h/2*sin(theta - np.pi/2))

    return [p0[0]+x, p1[0]+x, p2[0]+x, p3[0]+x], [p0[1]+y, p1[1]+y, p2[1]+y, p3[1]+y]


def draw_car(car_xs, car_ys, screen, screen_width, screen_height):
    p0 = (car_xs[0], car_ys[0])
    p1 = (car_xs[1], car_ys[1])
    p2 = (car_xs[2], car_ys[2])
    p3 = (car_xs[3], car_ys[3])

    p0 = pos_to_pix(*p0, screen_width, screen_height)
    p1 = pos_to_pix(*p1, screen_width, screen_height)
    p2 = pos_to_pix(*p2, screen_width, screen_height)
    p3 = pos_to_pix(*p3, screen_width, screen_height)

    pygame.draw.polygon(screen, BLUE, (p0, p1, p2, p3))

# @numba.njit
def get_car_parking_area_overlap(pa_xs, pa_ys, car_xs, car_ys):
    xs, ys = get_polygon_intersection_points(pa_xs, pa_ys, car_xs, car_ys)
    if xs != []:# Do something here to make this numba compatible
        xs, ys = polygon_from_points(xs, ys)
        intersection_area = get_area_of_polygon(xs, ys)
        car_area = get_area_of_polygon(car_xs, car_ys)
        return intersection_area / car_area

    return 0

def draw_line(x0, y0, x1, y1, screen, screen_width, screen_height):
    p0 = pos_to_pix(x0, y0, screen_width, screen_height)
    p1 = pos_to_pix(x1, y1, screen_width, screen_height)
    pygame.draw.line(screen, BLACK, p0, p1)

def draw_lines(xs, ys, screen, screen_width, screen_height):
    for i in range(len(xs)-1):
        draw_line(xs[i], ys[i], xs[i+1], ys[i+1], screen, screen_width, screen_height)

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

def draw_scan_res(scan_res, screen, screen_width, screen_height):
    for key in scan_res:
        d, l, p = scan_res[key]
        if p is not None:
            draw_line(l[0], l[1], p[0], p[1], screen, screen_width, screen_height)

@numba.njit
def clean_pa_scan_lines(pa_scan, obstacle_scan):
    for key in pa_scan:
        obs_d, obs_l, obs_p = obstacle_scan[key]
        pa_d, pa_l, pa_p = pa_scan[key]

        if pa_d is None:
            continue

        if pa_d > obs_d:
            pa_scan[key] = None, pa_l, None



if __name__ == '__main__':
    env = ParkingSimulator()
    env.reset()
    u1, u2 = 0, 0

    while True:
        for events in pygame.event.get():
            if events.type == pygame.QUIT:
                import sys
                sys.exit(0)
            elif events.type == pygame.KEYDOWN:
                if events.dict['unicode'] == 'w':
                    u1 = 1
                elif events.dict['unicode'] == 'a':
                    u2 = 1
                elif events.dict['unicode'] == 'd':
                    u2 = -1
                elif events.dict['unicode'] == 's':
                    u1 = -1
                elif events.dict['unicode'] == '\x1b': # esc
                    exit(0)
                elif events.dict['unicode'] == '\x1b': # esc
                    exit(0)
                elif events.dict['unicode'] == ' ':
                    env.reset()
            elif events.type == pygame.KEYUP:
                if events.dict['unicode'] == 'w':
                    u1 = 0
                elif events.dict['unicode'] == 'a':
                    u2 = 0
                elif events.dict['unicode'] == 'd':
                    u2 = 0
                elif events.dict['unicode'] == 's':
                    u1 = 0

        env.step((u1, u2))
        env.render(consume_events=False)

