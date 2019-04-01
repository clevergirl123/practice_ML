import numpy as np 
from sklearn.svm import SVC
import csv
import feature
import sift
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
def data_preprocess(file_path,voc):
    """
    file_path:the file that store the img info
    """
    with open(file_path,"rb") as file:
        reader = csv.reader(file)
        res_feature = []
        res_label = []
        for x in reader:
            print(x[0])
            res = feature.feature(x[0],voc)
            if(type(res) != type(np.zeros(0))):
                continue
            else:
                res_feature.append(feature.feature(x[0],voc))
                print(len(res_feature))
                res_label.append(int(x[1]))
    return res_feature,res_label
def evaluation(y, y_predict_prob):
    
    top5_correct_count = 0
    # print len(y_predict_prob)
    # print len(y_predict_prob[0])
    for idx, score in enumerate(y_predict_prob):
        sorted_index = np.argsort(-score).tolist()
        rank = sorted_index.index(y[idx])
        # print("123")
        if rank < 5:
            top5_correct_count += 1

    return (top5_correct_count+0.0)/len(y_predict_prob)


if __name__ == '__main__':

    voc = sift.init_sift("/home/fanfan/practice/svm_implement/basemodeldata/train.csv")
    with open("voc1","w") as f:
        pickle.dump(voc,f)
    X, y = data_preprocess("/home/fanfan/practice/svm_implement/basemodeldata/train.csv",voc) # 
    X_, y_ = data_preprocess("/home/fanfan/practice/svm_implement/basemodeldata/test.csv",voc) #
    with open("train_feature_base4f3s","w") as f:
        pickle.dump(X,f)
    with open("train_label_base4f3s","w") as f:
        pickle.dump(y,f)
    with open("test_feature_base4f3s","w") as f:
        pickle.dump(X_,f)
    with open("test_label_base4f3s","w") as f:
        pickle.dump(y_,f)
    # with open("train_feature_bases","r") as f:
    #     X = pickle.load(f)
    # with open("train_label_bases","r") as f:
    #     y = pickle.load(f)
    # with open("test_feature_bases","r") as f:
    #     X_ = pickle.load(f)
    # with open("test_label_bases","r") as f:
    #     y_ = pickle.load(f)

    


    # X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]) 
    # clf1 = MLPClassifier(hidden_layer_sizes=(50),solver = 'adam',alpha=0.5)
    clf = SVC(C = 0.8,kernel='linear',probability=True)
    # print clf 
    clf.fit(X, y)
    # clf1.fit(X,y)
    # with open('basemodel_4', 'w') as f:
    #     pickle.dump(clf, f)
    # with open("basemodel","r") as f:
    #     clf = pickle.load(f)

    # # X_ = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]]) 
    # # y_ = np.array([1, 1, 2, 2]) 

    y_predict_prob = clf.predict_log_proba(X_)
    # y_predict_prob1 = clf1.predict_log_proba(X_)
    print(y_predict_prob.shape)
    # print(y_predict_prob1.shape)
    # y_predict_prob2 = y_predict_prob*0.8+y_predict_prob1*0.1
    Top5_score = evaluation(y_, y_predict_prob)
    print('Top-5: %s '%(Top5_score))









