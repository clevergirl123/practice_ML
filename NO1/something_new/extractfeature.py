
from skimage.feature import hog
from skimage import feature
from PIL import Image
import numpy as np
from scipy.stats import zscore
import itertools



class LocalBinaryPatterns:
    def __init__(self,numPoints=8,radius=1):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self,image,eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image,self.numPoints,self.radius,method="uniform")
        (hist,_) = np.histogram(lbp.ravel(), bins = np.arange(0,self.numPoints+3),range = (0,self.numPoints+2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum()+eps)
        # return the histogram of Local Binary Patterns
        return hist

    def lbp_feature(self,arr,hnum=4,wnum=8):
        height,width = arr.shape
        phist = np.zeros(0)
        varr = np.vsplit(arr,hnum)
        for each in varr:
            harr = np.hsplit(each,wnum)
            for heach in harr:
                phist = np.append(phist,self.describe(heach))
            
        return phist

def hog_feature(image):
    fd = hog(image, orientations=16, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), visualise=False)
    return fd





def grayscale_comatrix(img_array):
    grayimg_array_8 = img_array 
    glcm = feature.greycomatrix(grayimg_array_8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    feature_vals = []
    featrue_params = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
    for param in featrue_params:
        feature_vals = np.append(feature_vals, feature.greycoprops(glcm, param))
    normalized_vals = zscore(feature_vals)
    return normalized_vals




def imgfeature(img_gray):
    res = []
    lbps = LocalBinaryPatterns()
    res = hog_feature(img_gray)*1
    res = np.append(res,grayscale_comatrix(img_gray)*0.75)
    res = np.append(res,lbps.lbp_feature(img_gray)*0.5)
    print(len(res))
    return res


def extractfeature(img_list):
    img_features = []
    i=0
    for item in img_list:
        i+=1
        img_feature = []
        # print(item[0])
        # print(i)
        img_feature=[imgfeature(item[0]),item[1]]
        img_features.append(img_feature)
    return img_features

# img = np.array(Image.open("./fox.jpg").convert("L"))

# print(img.shape)
# res = imgfeature(img)
# print(res)
# print(res.shape)
