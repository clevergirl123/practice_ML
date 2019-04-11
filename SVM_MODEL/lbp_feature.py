from skimage import feature
from PIL import Image
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, numPoints=8, radius=1):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist

    def lbp_feature(self, arr, hnum=4, wnum=8):
        height, width = arr.shape
        phist = np.zeros(0)
        varr = np.vsplit(arr, hnum)
        for each in varr:
            harr = np.hsplit(each, wnum)
            for heach in harr:
                phist = np.append(phist, self.describe(heach))

        return phist
