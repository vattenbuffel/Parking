from turtle import width
import numpy as np
from common_functions import BLACK
import pygame

def add_xy_and_offset(p, x, y, width, height):
    return p[0] + x + width/2, height/2 - y - 1 - p[1]

class Wheel:
    def __init__(self, w, h, x_offset_to_car, y_offset_to_car) -> None:
        self.alpha = np.arctan2(h/2, w/2)
        self.r = ((w/2)**2 + (h/2)**2)**0.5
        self.x_offset_to_car = x_offset_to_car
        self.y_offset_to_car = y_offset_to_car
        self.d_offset_to_car = (x_offset_to_car**2 + y_offset_to_car**2)**0.5
        self.alpha_offset_to_car = np.arctan2(y_offset_to_car, x_offset_to_car)

    def draw(self, car_x, car_y, theta, phi, screen):
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

        width, height = screen.get_width(), screen.get_height()
        p0 = add_xy_and_offset(p0, car_x+x_offset, car_y+y_offset, width, height)
        p1 = add_xy_and_offset(p1, car_x+x_offset, car_y+y_offset, width, height)
        p2 = add_xy_and_offset(p2, car_x+x_offset, car_y+y_offset, width, height)
        p3 = add_xy_and_offset(p3, car_x+x_offset, car_y+y_offset, width, height)

        pygame.draw.polygon(screen, BLACK, (p0, p1, p2, p3))
