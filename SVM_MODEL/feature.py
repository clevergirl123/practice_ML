import hog_interface as hog
import lbp_feature as lbp
import glcm
import numpy as np
import cv2
from PIL import Image


# import sift_feature as sift 
def img_fetch_resize(img_path,size = 64):
    img = cv2.imread(img_path)
    if(type(img) != type(np.zeros(0))):
        return None
    else:
        img_resize = cv2.resize(img, (size, size), interpolation = cv2.INTER_CUBIC)
        return img_resize


def feature(img_path):
    img_rgb = img_fetch_resize(img_path)
    if (type(img_rgb) != type(np.zeros(0))):
        return None
    else:
        img_gray = cv2.cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        lbps = lbp.LocalBinaryPatterns()
        res = hog.hog_feature(img_gray) * 1.15
        res = np.append(res, glcm.grayscale_comatrix(img_gray))
        # res = np.append(res,color.color_feature(img_gray))
        res = np.append(res, lbps.lbp_feature(img_gray) * 0.85)
        # res = lbps.lbp_feature(img_gray)

    return res
    # image = np.array(Image.open("/home/fanfan/practice/feature/fox.jpg").convert("L"))
    # # print(image.size)
    # fd = feature("/home/fanfan/practice/feature/fox.jpg")
    # print(fd.size)
    # print(fd)
    # print(type(fd))
