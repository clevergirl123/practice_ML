import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog


class HogFit:

    def __init__(self, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L1'):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def hog_feature(self, image):
        fd = hog(image, self.orientations, self.pixels_per_cell, self.cells_per_block, visualise=False)
        return fd

    def add_orient(self, step=2):
        if self.orientations < 120:
            self.orientations += step
        else:
            self.orientations = 48

    def del_orient(self, step=2):
        if self.orientations > 4:
            self.orientations -= step
        else:
            self.orientations = 8

    def add_pixels(self, step_0=1, step_1=1):
        if self.pixels_per_cell[0] < (64 - step_0):
            temp_0 = self.pixels_per_cell[0] + step_0
        else:
            temp_0 = 32
        if self.pixels_per_cell[1] < (64 - step_1):
            temp_1 = self.pixels_per_cell[1] + step_1
        else:
            temp_1 = 32
        self.pixels_per_cell = (temp_0, temp_1)

    def del_pixels(self, step_0=1, step_1=1):
        if self.pixels_per_cell[0] > (4 + step_0):
            temp_0 = self.pixels_per_cell[0] - step_0
        else:
            temp_0 = 4
        if self.pixels_per_cell[1] > (4 + step_1):
            temp_1 = self.pixels_per_cell[1] - step_1
        else:
            temp_1 = 4
        self.pixels_per_cell = (temp_0, temp_1)

    def add_cells(self, step_0=1, step_1=1):
        if self.cells_per_block[0] < (self.pixels_per_cell[0] - step_0):
            temp_0 = self.cells_per_block[0] + step_0
        else:
            temp_0 = 4
        if self.cells_per_block[1] < (self.pixels_per_cell[1] - step_1):
            temp_1 = self.cells_per_block[1] + step_1
        else:
            temp_1 = 4
        self.pixels_per_cell = (temp_0, temp_1)

    def del_cells(self, step_0=1, step_1=1):
        if self.cells_per_block[0] > (2 + step_0):
            temp_0 = self.cells_per_block[0] - step_0
        else:
            temp_0 = 2
        if self.cells_per_block[1] > (2 + step_1):
            temp_1 = self.cells_per_block[1] - step_1
        else:
            temp_1 = 2
        self.cells_per_block = (temp_0, temp_1)
