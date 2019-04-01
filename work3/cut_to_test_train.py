import numpy as np 
import cv2
import csv
import os
import uuid
from sklearn.model_selection import train_test_split
#def cut_img():

def loaddata(img_path, file_path, out_img_path, out_file_path):
    with open(out_file_path, 'w') as f:
        f.write('')
    result = []
    pictures = list(csv.reader(open(file_path, 'r')))
    # print(pictures)
    count = 0
    for item in pictures:
        count += 1
        big_img = cv2.imread(os.path.join(img_path, item[0]))
        # print(img)
        small_img = big_img[int(item[3]):int(item[5]), int(item[2]):int(item[4])]
        img_name = str(uuid.uuid1()) + '.jpg'
#         print(img_name)
        cv2.imwrite(os.path.join(out_img_path, img_name), small_img)
        result.append((small_img, item[1]))
        with open(out_file_path, 'a+') as f:
            f.write(img_name + ',' + str(item[1]) + '\n')
    # print(result)
    print("加载完毕，哈哈哈哈哈！")
    return result

# 划分训练集和测试集
def data_split(file_path, train_path, test_path):
    with open(test_path, 'w') as f:
        f.write('')
    with open(train_path, 'w') as f:
        f.write('')
    pictures = np.array(list(csv.reader(open(file_path, 'r'))))
    print(type(pictures))
    x_train,x_test,y_train,y_test = train_test_split(pictures[:, :1], pictures[:, 1],test_size=0.3,random_state=0)
    for idx, name in enumerate(x_train):
        with open(train_path, 'a+') as f:
            f.write(name[0] + ',' + y_train[idx] + '\n')
    for idx, name in enumerate(x_test):
        with open(test_path, 'a+') as f:
            f.write(name[0] + ',' + y_test[idx] + '\n')
    print('恭喜你，划分完毕！')

if __name__ == '__main__':
    print("好戏开始！！！！！！！")
    img_path = '/home/aistudio/data/data274/train/train'
    file_path = '/home/aistudio/data/data274/train/train.txt'
    out_img_path = '/home/aistudio/data/data274/train/train_new'
    out_file_path = '/home/aistudio/data/data274/train/train_new.csv'
    train_file_path = '/home/aistudio/data/data274/train/train.csv'
    test_file_path = '/home/aistudio/data/data274/train/test.csv'
    loaddata(img_path, file_path, out_img_path, out_file_path)
    data_split(out_file_path, train_file_path, test_file_path)
