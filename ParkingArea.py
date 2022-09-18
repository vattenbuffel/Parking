import pygame
from common_functions import pos_to_pix, GREEN

class ParkingArea:
    def __init__(self, xs, ys) -> None:
        self.xs = xs
        self.ys = ys

    def draw(self, screen):
        ps = tuple((pos_to_pix(self.xs[i], self.ys[i], screen.get_width(), screen.get_height())) for i in range(len(self.xs)) )
        pygame.draw.polygon(screen, GREEN, ps)