import os
import cv2
import numpy as np


def loaddata(fpath):
    label = {"bear": 0, "bird": 1, "car": 2, "cow": 3, "elk": 4, "fox": 5, "giraffe": 6, "horse": 7, "koala": 8, "lion": 9, "monkey": 10,
             "plane": 11, "puppy": 12, "sheep": 13, "statue": 14, "tiger": 15, "tower": 16, "train": 17, "whale": 18, "zebra": 19, "bicycle": 20}

    info_list = []

    for root, dirs, files in os.walk(fpath):
        for dir in dirs:
            filelist = os.listdir(os.path.join(root, dir))
            for file in filelist:
                img = cv2.imread(os.path.join(root, dir, file))

                if(type(img) != type(np.zeros(0))):
                    continue
                else:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img_64 = cv2.resize(img_gray, (64, 64),
                                        interpolation=cv2.INTER_CUBIC)

                    info_list.append([img_64, label.get(dir)])
    return info_list


# a = loaddata('/home/frank/ML/ds2018')
# print(len(a))
