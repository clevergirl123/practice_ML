import numpy as np
from sklearn.svm import SVC
import csv
import feature
from sklearn.decomposition import PCA
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def data_preprocess(file_path):
    """
    file_path:the file that store the img info
    """
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        res_feature = []
        res_label = []
        for x in reader:
            # print(x[0])
            res = feature.feature(x[0])
            if (type(res) != type(np.zeros(0))):
                continue
            else:
                res_feature.append(feature.feature(x[0]))
                # print(len(res_feature))
                res_label.append(int(x[1]))
    return res_feature, res_label


def evaluation(y, y_predict_prob):
    top5_correct_count = 0
    # print len(y_predict_prob)
    # print len(y_predict_prob[0])
    # print('----------- debug : y_predict_prob ---------------')
    # print(np.array(y_predict_prob).shape)
    result_all = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                  16: 0, 17: 0, 18: 0, 19: 0, 20: 0}
    result_n = [
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
         17: 0, 18: 0, 19: 0, 20: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
         17: 0, 18: 0, 19: 0, 20: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
         17: 0, 18: 0, 19: 0, 20: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
         17: 0, 18: 0, 19: 0, 20: 0},
        {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
         17: 0, 18: 0, 19: 0, 20: 0}]
    for idx, score in enumerate(y_predict_prob):
        # print('----------- debug : score ---------------')
        # print(np.array(score).shape)
        # print(score)
        sorted_index = np.argsort(-score).tolist()
        rank = sorted_index.index(y[idx])
        # print("123")
        if rank < 5:
            top5_correct_count += 1
            top_n = 0
            for i in np.argsort(score)[-5:][::-1]:
                result_n[top_n][i] += 1
                result_all[i] += 1
                top_n += 1
    print(result_all)
    print('-------------------------------------------')
    for items in result_n:
        print(items)
    return (top5_correct_count + 0.0) / len(y_predict_prob)


if __name__ == '__main__':
    print('starting')
    X, y = data_preprocess(r"F:\Practice\new_20180725\train.csv")  #
    print('训练集读取完成')
    X_, y_ = data_preprocess(r"F:\Practice\new_20180725\test.csv")  #
    print("测试集读取完成")
    print('开始训练模型')

    # estimator = PCA(n_components=2000)
    # pca_x_train = estimator.fit_transform(X)
    # pca_x_test = estimator.transform(X_)

    clf = SVC(C=0.8, kernel='linear', probability=True)
    # clf.fit(pca_x_train, y)
    clf.fit(X, y)
    with open("model.model", 'wb') as f:
        pickle.dump(clf, f)
    with open("model.model", 'rb') as f:
        clf = pickle.load(f)
    print('模型训练完成，  开始预测。  ')
    # y_predict_prob = clf.predict_log_proba(pca_x_test)
    y_predict_prob = clf.predict_log_proba(X_)
    Top5_score = evaluation(y_, y_predict_prob)
    print("clf = SVC(C=0.8, kernel='linear', probability=True)")
    print("hog(image, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=False)")
    print('hog : 1.15,  lbp : 0.85,  glcm : 1')
    print('Top-5: %s ' % (Top5_score))
