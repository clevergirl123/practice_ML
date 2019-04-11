from skimage import feature
import cv2
from PIL import Image
import numpy as np
import itertools
from scipy.stats import zscore

def grayscale_comatrix(img_array):
	# grayimg_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
	grayimg_array_8 = img_array 
	glcm = feature.greycomatrix(grayimg_array_8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
	feature_vals = []
	featrue_params = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
	for param in featrue_params:
		feature_vals = np.append(feature_vals, feature.greycoprops(glcm, param))
	normalized_vals = zscore(feature_vals)
	return normalized_vals

# image = np.array(Image.open("./fox.jpg").convert("L"))
# print(image.size)
# fd = grayscale_comatrix(image)
# print(fd)
# print(fd.size)
# print(type(fd))

