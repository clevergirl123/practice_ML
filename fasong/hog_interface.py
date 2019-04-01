

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import hog

from skimage import data, exposure

def hog_feature(image):
    # image = np.array(Image.open(r'F:\Practice\neu-dataset\fox\fox246.jpg'))
    fd = hog(image, orientations=12, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualise=False)
    return fd
# image = np.array(Image.open("./fox.jpg").convert("L"))
# print(image.size)
# fd = hog_feature(image)
# print(fd)
# print(fd.size)
# print(type(fd))
