import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, tan
import pygame

def model(x, u1, u2, dt, L):
    # x is state vector [x, y, theta, phi, dx, dy, dtheta, dphi]
    # u1 is acc, u2 is steering acc

    theta = x[2, 0]
    phi = x[3, 0]
    dx = x[4, 0]
    dy = x[5, 0]
    dtheta = x[6, 0]
    dphi = x[7, 0]

    v = np.array([
        dx,
        dy,
        dtheta, 
        dphi,
        cos(theta)*u1,
        sin(theta)*u1,
        1/L*tan(phi)*u1,
        u2,
    ]).reshape(8, 1)

    x1 = x + v*dt
    # Limit dtheta
    x1[6] = np.clip(x1[6], -np.deg2rad(10), np.deg2rad(10))
    
    #limit phi and dphi
    x1[7] = np.clip(x1[7], -np.deg2rad(30), np.deg2rad(30))
    x1[3] = np.clip(x1[3], -np.deg2rad(45), np.deg2rad(45))

    return x1


def pos_to_pix(x, y):
    x += width/2
    y = height - y - 1
    y -= height/2
    return x, y

def blitRotate(surf, image, pos, originPos, angle):
    # offset from pivot to center
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
    
    # roatated offset from pivot to center
    rotated_offset = offset_center_to_pivot.rotate(-angle)

    # roatetd image center
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)

    # get a rotated image
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)

    # rotate and blit the image
    surf.blit(rotated_image, rotated_image_rect)
  
    # draw rectangle around the image
    pygame.draw.rect(surf, (255, 0, 0), (*rotated_image_rect.topleft, *rotated_image.get_size()),2)

def draw_car(x, y, theta):
    h = car_height
    w = car_width
    theta0 = np.arctan2(h/2, w)   #angle at which p1 starts at
    d = ((h/2)**2 + (w)**2)**0.5
    p0 = (h/2*cos(theta + np.pi/2), h/2*sin(theta + np.pi/2))
    p1 = (d * cos(theta + theta0), d * sin(theta + theta0))
    p2 = (d * cos(theta - theta0), d * sin(theta - theta0))
    p3 = (h/2*cos(theta - np.pi/2), h/2*sin(theta - np.pi/2))

    def add_xy_and_offset(p, x, y):
        return p[0] + x + width/2, height/2 - y - 1 - p[1]

    p0 = add_xy_and_offset(p0, x, y)
    p1 = add_xy_and_offset(p1, x, y)
    p2 = add_xy_and_offset(p2, x, y)
    p3 = add_xy_and_offset(p3, x, y)

    pygame.draw.polygon(screen, BLUE, (p0, p1, p2, p3))


    surface = pygame.Surface(size)
    pygame.draw.polygon(surface, GREEN, ((0, 100), (0, 200), (200, 200), (200, 300), (300, 150), (200, 0), (200, 100)))
    surface = pygame.transform.rotate(surface, np.rad2deg(theta))
    surface = pygame.transform.scale(surface, (arrow_size, arrow_size))
    p = pos_to_pix(x_, y)
    p = (p[0] - arrow_size/2, p[1] - arrow_size/2 )
    screen.blit(surface, p)


x = np.zeros((8,1))
x[2] = 1.57/2
u1 = 0
u2 = 0.0
dt = 0.001
L = 200

width, height = (1280, 1280)
arrow_size = 16
car_width = L
car_height = 50
screen = pygame.display.set_mode((width,height))
pygame.font.init()
my_font = pygame.font.SysFont(None, 30)

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255, 0 ,0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
size = (300, 300)

w_p, a_p, d_p, s_p = False, False, False, False

while True:
    screen.fill(WHITE)
    x_ = x[0, 0]
    y = x[1, 0]
    theta = x[2, 0]
    phi = x[3, 0]
    dx = x[4, 0]
    dy = x[5, 0]
    dtheta = x[6, 0]
    dphi = x[7, 0]

    draw_car(x_, y, theta)

    text_surface = my_font.render(f"phi: {phi:.2f}, dphi: {dphi:.2f}, dtheta: {dtheta:.2f}, u1: {u1:.2f}, u2: {u2:.2f}", True, BLACK)
    screen.blit(text_surface, (50,50))

    pygame.display.update()

    for events in pygame.event.get():
        if events.type == pygame.QUIT:
            import sys
            sys.exit(0)
        elif events.type == pygame.KEYDOWN:
            if events.dict['unicode'] == 'w':
                w_p = True
                u1 = 1
            elif events.dict['unicode'] == 'a':
                a_p = True
                u2 = np.deg2rad(2)
            elif events.dict['unicode'] == 'd':
                d_p = True
                u2 = np.deg2rad(-2)
            elif events.dict['unicode'] == 's':
                s_p = True
                u1 = -1
            elif events.dict['unicode'] == '\x1b': # esc
                exit(0)
            elif events.dict['unicode'] == '\x1b': # esc
                exit(0)
            elif events.dict['unicode'] == ' ':
                x = np.zeros((8,1))
        elif events.type == pygame.KEYUP:
            if events.dict['unicode'] == 'w':
                w_p = False
            elif events.dict['unicode'] == 'a':
                a_p = False
            elif events.dict['unicode'] == 'd':
                d_p = False
            elif events.dict['unicode'] == 's':
                s_p = False

    if not w_p and not s_p:
        # Break a bit
        u1 = -(dx**2 + dy**2)**0.5
    if not a_p and not d_p:
        # Break a bit
        u2 = -dphi


    for i in range(100):
        x = model(x, u1, u2, dt, L)



