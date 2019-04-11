import numpy as np
from sklearn.svm import SVC
import csv
import feature
import os
import glob
from sklearn.decomposition import PCA
import pickle
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

label_map_1 = {0: 'bear',
               1: 'bicycle',
               2: 'bird',
               3: 'car',
               4: 'cow',
               5: 'elk',
               6: 'fox',
               7: 'giraffe',
               8: 'horse',
               9: 'koala',
               10: 'lion',
               11: 'monkey',
               12: 'plane',
               13: 'puppy',
               14: 'sheep',
               15: 'statue',
               16: 'tiger',
               17: 'tower',
               18: 'train',
               19: 'whale',
               20: 'zebra'}

label_map_20 = {0: 'bear',
                1: 'bird',
                2: 'car',
                3: 'cow',
                4: 'elk',
                5: 'fox',
                6: 'giraffe',
                7: 'horse',
                8: 'koala',
                9: 'lion',
                10: 'monkey',
                11: 'plane',
                12: 'puppy',
                13: 'sheep',
                14: 'statue',
                15: 'tiger',
                16: 'tower',
                17: 'train',
                18: 'whale',
                19: 'zebra',
                20: 'bicycle'}


def testmodel(input_img, model_path='group_6.model', top=5):
    result = []
    img_data = []
    img_tag = []
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    for img in input_img:
        img_data.append(img[0])
        img_tag.append(img[1])
    predict_prob = clf.predict_log_proba(img_data)
    for idx, score in enumerate(predict_prob):
        sorted_index = np.argsort(-score).tolist()
        rank = sorted_index.index(img_tag[idx])
        if rank < top:
            result.append(True)
        else:
            result.append(False)
    print('Top-5: ' + str(sum(rank)/len(rank)))
    return result


def tran(root_path, image, image_name):
    files = os.listdir(root_path)
    for file in files:
        path = os.path.join(root_path, file)
        if os.path.isdir(path):
            tran(path, image, image_name)
        else:
            if path[-4:] == '.jpg':
                res = feature.feature(path)
                if isinstance(res, type(np.zeros(0))):
                    image.append(res)
                    image_name.append(path)
    return image, image_name


def to_predict(image_root_path, model_path, out_path='result.csv', top=5):
    temp_img = []
    temp_name = []
    result = []
    image, img_name = tran(image_root_path, temp_img, temp_name)
    print('--------- Data read complete! ---------------------')
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    predict_prob = clf.predict_log_proba(image)
    for idx, score in enumerate(predict_prob):
        temp_result = img_name[idx]
        sorted_index = np.argsort(-score).tolist()
        for item in sorted_index[:top]:
            temp_result = temp_result + ',' + label_map_1[item]
        print(temp_result)
        temp_result += '\n'
        result.append(temp_result)
        with open(out_path, 'a+') as f:
            f.write(temp_result)


if __name__ == '__main__':
    image_path = r'C:\Users\LQB\Desktop\train'
    model_path = r'F:\学习\大三\大三下\生产实习\SVM_MODEL\model.model'
    out_path = r'F:\学习\大三\大三下\生产实习\SVM_MODEL\result_' + str(int(time.time())) + '.csv'
    with open(out_path, 'w') as f:
        f.write('')
    to_predict(image_path, model_path, out_path)
