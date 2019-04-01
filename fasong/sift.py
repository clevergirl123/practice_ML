import cv2
import csv
import numpy as np
from scipy.cluster.vq import *

def des2feature(centures,img,num_words=30):
    # img = cv2.imread(img)
    des = sift_feature(img)
    img_feature_vec = np.zeros((1, num_words), 'float32')
    for i in range(des.shape[0]):
        feature_k_rows = np.ones((num_words, 128), 'float32')
        feature = des[i]
        #print(centures.shape)
        #print(feature.shape)
        #print(feature_k_rows.shape)
        feature_k_rows = feature_k_rows * feature
        feature_k_rows = np.sum((feature_k_rows - centures) ** 2, 1)

        index = np.argmax(feature_k_rows)
        img_feature_vec[0][index] += 1
    img_feature_vec = img_feature_vec.flatten()
    return img_feature_vec


def sift_feature(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(img)
    keypoints, descriptor = sift.compute(img, keypoints)
    return descriptor


def init_sift(train_file,size=64,bagnum = 30):
    # descriptors = []
    des = None
    # print(train_file)
    # des = np.zeros(0)
    with open(train_file,"r") as train_file:
        reader = csv.reader(train_file)
        for item in reader:
            
            img = cv2.imread(item[0])
            # print(img)
            if img is not None:
                print(item[0])
                print("hhh")
                img = cv2.resize(img, (size, size), interpolation = cv2.INTER_CUBIC)
                if des is not None:
                    des = np.vstack((des,sift_feature(img)))
                else:
                    des = sift_feature(img)
            # print(item[0])
    print(des)
    whitened = whiten(des)

    voc,variance = kmeans(whitened,bagnum,2)

    return voc

# voc = init_sift("/home/fanfan/practice/test.csv")
# print(voc)
# print(voc.shape)
# res = des2feature(voc,"./fox.jpg")
# print(res)
# print(res.shape)

