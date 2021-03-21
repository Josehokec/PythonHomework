"""
McTwo algorithm
@author : Liu Shizhe
Accuracy = (TP + TN) / (TP + TN + FP + FN)
30 times circulation
"""

from csv import reader
import numpy as np
from minepy import MINE
import pandas as pd
from copy import deepcopy
import configparser

from sklearn.model_selection import cross_val_score

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import neighbors as nb
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.model_selection import KFold,StratifiedKFold

import warnings

warnings.filterwarnings("ignore")  # 程序运行时sklearn某一个模块有警告，忽略警告


def get_data(filename, positive_name):
    """
    功能：读取指定TXT文件的数据
    参数：TXT文件的文件名以及阳性样本的名称
    函数返回: 数据矩阵Matrix和类标签Class
    ***说明：TXT文件存储格式是特征/基因按行存储，样本按列存储
    ***我喜欢的存储格式是行是样本，列是特征，所以我转置了TXT文件的数据
    """
    with open(filename, 'rt') as raw_data:
        readers = reader(raw_data, delimiter='\t')
        x = list(readers)
        data = np.array(x)  # 换成矩阵能加快运算速度
        data = data.T  # 转置一下

    # 现在data的每行是一个样本，每列是一个特征，最后一列是类标签
    sample_name = data[1:, 0]                       # 样本名字，如'Normal'、'Tumor'
    label_C = [0.0] * len(sample_name)              # label_class大小等于样本数量

    for i in range(len(sample_name)):
        if sample_name[i] == positive_name:
            label_C[i] = 1.0                        # 如果该样本是阳性的，置成1，否则保持0不变

    Class = np.array(label_C)                       # 统一用numpy处理数据，C_value就是mic的C
    Matrix = data[1:, 1:]                           # 标准的数据矩阵，删除了第一列和第一行；一行是一个样本，一列是一个特征
    Matrix = Matrix.astype(np.float64)              # 如果不转换类型，matrix是字符串的矩阵

    return Matrix, Class


def McOne(Matrix, C, r):
    """
    功能：McOne算法==>初步筛选出相关特征
    参数：Matrix：数据矩阵 C：类标签，即0/1矩阵 r：阈值 
    返回：第一步筛选出的特征矩阵FReduce
    """

    # Step one : initialization
    k = Matrix.shape[1]                             # k的值应该等于二维数组列数
    micFC = [0 for i in range(k)]                   # 创建一个micFC数组，等效于micFC = [0] * len(data)
    Subset = [0 for i in range(k)]                  # 创建Subset数组

    numSubset = 0                                   # 记录子集个数 算法伪代码是1，这里置0最合适
    mine = MINE(alpha=0.6, c=15, est="mic_approx")  # 引入MIC

    # Step two : screen | if >= r then be selected
    for i in range(k):                              # 从0~(k - 1)循坏
        mine.compute_score(Matrix[:, i], C)
        micFC[i] = mine.mic()                       # 计算第i个特征跟标签C的MIC值，存入micFC数组中
        if micFC[i] >= r:                           # 如果大于阈值可以认为是一类
            Subset[numSubset] = i
            numSubset = numSubset + 1
    # print('k', k)
    # print('numSubset', numSubset)
    # Step three : descending sort 
    # 思想：(x1, y1) > (x2, y2)成立建立在x1 > x2 || (x1 == x2 &&  y1 > y2)条件上
    arr = [(micFC[Subset[i]], Subset[i]) for i in range(numSubset)]
    Subset = [var for _, var in sorted(arr, reverse=True)]

    # Step four : delete redundancy
    # 思想：mic(e,q) >= mic(q,c)意味着特征e和特征q冗余，删除特征q
    e = 0
    while e < numSubset:
        q = e + 1
        while q <= numSubset - 1:                   # 特别注意减1
            mine.compute_score(Matrix[:, Subset[e]], Matrix[:, Subset[q]])
            temp = mine.mic()
            if temp >= micFC[Subset[q]]:
                # 如果条件成立，那么删除冗余特征，数组移动，可以优化
                for i in range(q, numSubset - 1):
                    Subset[i] = Subset[i + 1]
                numSubset = numSubset - 1
            else:
                q = q + 1
        e = e + 1

    new_Subset = Subset[0: numSubset]  # 删除冗余后新的子集
    FReduce = Matrix[:, new_Subset]  # 筛选后的特征子集

    return FReduce


def best_first_search(X, y, max_counter):
    """
    功能：McTwo第二步==>使用改进的BFS进一步减少特征
    参数：X为McOne函数的返回值；y是类标签，其元素取值是{0, 1}，max_counter最大扩充结点
    返回：最终筛选出来的特征矩阵，即McTwo算法筛选出来的结果
    注意：需要用到KNN算法来做进一步的筛选，调用Python相关包即可
    """

    feature_num = X.shape[1]                    # 特征的数量 列的数量
    counter = 0                                 # 扩充3个节点准确率还是没有提升，则停止
    nn = nb.KNeighborsClassifier(n_neighbors=1)
    final_feature = []                          # 保存的是最后要选的特征的索引
    max_accuracy = 0

    while counter < max_counter:                # 3是试验值
        current_feature = []                    # 当前的特征索引数组
        current_max_accuracy = 0                # 当前最大的准确度
        for i in range(0, feature_num):         # 遍历所有的特征
            if i not in final_feature:
                # 注意：需要深拷贝，否则会引起final_feature改变
                now_feature = deepcopy(final_feature)
                now_feature.append(i)
                now_score = cross_val_score(nn, X[:, now_feature], y, cv=5)
                # 如果扩充一个特征发现acc上升了，那么就选择这个特征
                if now_score.mean() > current_max_accuracy:
                    current_max_accuracy = now_score.mean()
                    current_feature = now_feature
        # 如果没有结点可以扩展那就跳出循环 
        if len(final_feature) == feature_num:
            break

        final_feature = current_feature
        if current_max_accuracy > max_accuracy:
            # 最大准确度提高了，最大准确度更新，计数器置0
            max_accuracy = current_max_accuracy
            counter = 0
        else:
            counter = counter + 1               # 准确度没有提高，计数器加1

    if counter > 0:                             # 把没引起准确度提升的特征删除
        for i in range(0, counter):
            final_feature.pop()

    return X[:, final_feature]


def counter_TN_TP_FN_FP(X_pred, y_test):
    """
    功能：统计当前结果TN、TP、FN、FP的值
    TP : True positive | FP : False positive | FN : False negative | TN : True negative
    参数：预测的类别和实际的类别
    返回：TN, TP, FN, FP, 样本长度
    """
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_test)):
        if X_pred[i] == 1 and y_test[i] == 1:
            TP = TP + 1
        elif X_pred[i] == 1 and y_test[i] == 0:
            FP = FP + 1
        elif X_pred[i] == 0 and y_test[i] == 1:
            FN = FN + 1
        elif X_pred[i] == 0 and y_test[i] == 0:
            TN = TN + 1

    return TN, TP, FN, FP


def cal_result(TP, TN, FP, FN):
    """
    功能：计算预测结果的Sn, Sp, Acc, Avc, MCC
    参数：TP TN FP FN
    返回：Sn, Sp, Acc, Avc, MCC
    注意MCC的坟墓可能会碰到0，与需要做特判
    """
    Sn = TP / (TP + FN)
    Sp = TN / (TN + FP)
    Acc = (TP + TN) / (TP + TN + FN + FP)
    Avc = (Sn + Sp) / 2
    temp = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    if temp == 0.0:
        MCC = 0
    else:
        MCC = (TP * TN - FP * FN) / temp
    return Sn, Sp, Acc, Avc, MCC


"""
这一步的工作是读配置文件的内容
"""
cf = configparser.ConfigParser()                # 得到配置解释器句柄         
cf.read('config.ini')                           # 读取配置文件
filelen = cf.get('file', 'fileslength')         # 得到文件名字
filenames = list()
positive_names = list()
for i in range( int(filelen)):
    cur_filename = 'filename' + str(i + 1)
    cur_positive_name = 'pos_name' + str(i + 1)
    # print(cur_filename)
    filename_str = cf.get('file', cur_filename)               # 得到文件名字
    positive_name_str = cf.get('file', cur_positive_name)     # 得到文件名字
    filenames.append(filename_str)
    positive_names.append(positive_name_str)

r_str = cf.get('const', 'r')
counter_str = cf.get('const', 'bfs_counter')
r = float(r_str)                                  # McOne第一步需要的阈值
counter = int(counter_str)                  # bfs最大扩展结点数量

"""
开始测试数据集
这里我直接测试所有文件，一次性得到结果
"""
for i in range( int(filelen) ):
    print('\nfilename : ', filenames[i])
    Matrix, C = get_data(filenames[i], positive_names[i])
    StepOneMatrix = McOne(Matrix, C, r)
    ResultMatrix = best_first_search(StepOneMatrix, C, counter)
    print('Feature number - McOne :', StepOneMatrix.shape[1], ' McTwo : ', ResultMatrix.shape[1])

    X, y = ResultMatrix, C

    Svm = svm.SVC()
    Nbayes = GaussianNB()
    Dtree = tree.DecisionTreeClassifier()
    Knn = KNeighborsClassifier(n_neighbors=1)
    clfs = [Svm, Nbayes, Dtree, Knn]
    clfs_name = ['SVM', 'NBayes', 'DTree', 'NN']
    clf_results = list()

    for clf in clfs:
        times = 30                              # 随机30次
        Sns = []                                # Sn = TP / (TP + FN) 阳性样本预测对的
        Sps = []                                # Sp = TN / (TN + FP) 阴性样本预测对的
        Accs = []                               # Acc = (TP + TN) / (TP + TN + FN + FP)
        Avcs = []                               # Avc = (Sn + Sp) / 2
        MCCs = []                               # MCC = baidu

        for k in range(times):
            TNs = []
            TPs = []
            FNs = []
            FPs = []
            # 论文指明需要5fold
            kf = StratifiedKFold(n_splits = 5, shuffle = True)
            for train, test in kf.split(X, y):
                # fit是训练分类器，predict是预测
                clf.fit(X[train, :], y[train])
                X_pred = clf.predict(X[test, :])
                tn, tp, fn, fp = counter_TN_TP_FN_FP(X_pred, y[test])
                TNs.append(tn)
                TPs.append(tp)
                FNs.append(fn)
                FPs.append(fp)

            Sn, Sp, Acc, Avc, MCC = cal_result(sum(TPs), sum(TNs), sum(FPs), sum(FNs))

            Sns.append(Sn)
            Sps.append(Sp)
            Accs.append(Acc)
            Avcs.append(Avc)
            MCCs.append(MCC)

        average_sn = np.mean(Sns)
        average_sp = np.mean(Sps)
        average_acc = np.mean(Accs)
        average_avc = np.mean(Avcs)
        average_mcc = np.mean(MCCs)

        results = []  # 某一个分类器的五个值的结果列表
        results.append(average_sn)
        results.append(average_sp)
        results.append(average_acc)
        results.append(average_avc)
        results.append(average_mcc)
        clf_results.append(results)

    print('Data : Sn | Sp | Acc | Avc | MCC')
    for i in range(len(clf_results)):
        print(clfs_name[i], ' : ', clf_results[i])
