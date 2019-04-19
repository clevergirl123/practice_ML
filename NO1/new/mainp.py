import extractfeature
import loaddata
import trainmodel
import testmodel
import time
import numpy as np

if __name__ == '__main__':
    t0 = time.time()
    train_path = "F:/Practice/store_tag/train/10783/train.csv"
    test_path = "F:/Practice/store_tag/train/10783/test.csv"
    train_list = loaddata.loaddata(train_path)
    print("train_list")
    test_list = loaddata.loaddata(test_path)
    print("test_list")
    train_feature = extractfeature.extractfeature(train_list)
    print("train_feature")
    test_feature = extractfeature.extractfeature(test_list)
    print("test_feature")
    # print(test_feature)
    # print(type(test_feature))
    # print(np.array(test_feature).shape)
    trainmodel.trainmodel(train_feature)
    print("train finished")
    #testmodel.testmodel(input_img=test_feature,top=2)
    print('--------------------top2')
    testmodel.testmodel(input_img=test_feature,top=5)
    print("in_C=1, in_kernel='linear'")
    print('hog:1 glcm:1 lbp:1')
    print('--------------------top5')
    print(time.time()-t0)
