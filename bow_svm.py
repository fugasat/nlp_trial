import os
import codecs
import numpy as np
import MeCab
import pickle
import random
import time
from gensim import corpora, models
from sklearn import svm, grid_search
import dataset


if __name__ == '__main__':
    # 教師データ作成
    data_train, label_train, data_test, label_test = dataset.create_dataset()

    # Scikit-learnのSVMアルゴリズムを使ってトレーニング
    svc = svm.SVC()
    # コストパラメータの設定（Grid Search用に複数設定）
    cs = [0.001, 0.01, 0.1, 1, 10]
    # RBFカーネルパラメータの設定（Grid Search用に複数設定）
    gammas = [0.001, 0.01, 0.1, 1]
    parameters = {'kernel': ['rbf'], 'C': cs, 'gamma': gammas}
    clf = grid_search.GridSearchCV(svc, parameters)
    # data_trainは、BOWでベクトル化した各文書のリスト
    # label_trainは、文書のカテゴリのリスト（ラベル）
    start = time.time()
    print("training...")
    clf.fit(data_train, label_train)

    # 学習した結果のスコアの確認
    # (memo)
    # train=100 : mean: 0.74889, std: 0.00338
    # train=200 : mean: 0.79056, std: 0.02695
    # train=300 : mean: 0.75852, std: 0.04206
    # train=90% : mean: 0.12215, std: 0.00007
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time / 60) + "[min]")
    print('Grid Score: %s' % clf.grid_scores_)

    # テストデータの実行
    # Grid Searchを実行した場合、一番スコアの高いハイパーパラメータを使ったモデルでスコアを出してくれる
    # (memo)
    # train=100 : Score: 0.810376775788
    # train=200 : Score: 0.875179340029
    # train=300 : Score: 0.898631308811
    # train=90% : Score: 0.955704697987
    start = time.time()
    print("check test data...")
    score = clf.score(data_test, label_test)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time / 60) + "[min]")
    print('Test Score: ' + str(score))

    pass