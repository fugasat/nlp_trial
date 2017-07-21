import random
import numpy as np
import time
import pickle
import tensorflow as tf
import dataset


if __name__ == '__main__':
    # 入力データ（300次元の文書データ）
    x = tf.placeholder(tf.float32, [None, 300])
    # 正解ラベル（文書カテゴリのOne Hotベクトル）
    y_ = tf.placeholder(tf.float32, [None, 9])

    # 変数の用意（Weight、Bias）
    # 隠れ層のWeight
    W_h = tf.Variable(tf.random_normal([300, 200], mean=0.0, stddev=0.05))
    # 出力層のWeight
    W_o = tf.Variable(tf.random_normal([200, 9], mean=0.0, stddev=0.05))
    # 隠れ層のバイアス
    b_h = tf.Variable(tf.zeros([200]))
    # 出力層のバイアス
    b_o = tf.Variable(tf.zeros([9]))

    # 隠れ層（シグモイド関数）
    h = tf.sigmoid(tf.matmul(x, W_h) + b_h)
    # 出力層（ソフトマックス）
    y = tf.nn.softmax(tf.matmul(h, W_o) + b_o)

    # 交差エントロピー
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # 正則化（Regularization）
    l2 = tf.nn.l2_loss(W_h) + tf.nn.l2_loss(W_o)
    l2_lambda = 0.01
    # コスト関数
    loss = cross_entropy + l2_lambda * l2

    # 勾配降下法により最適解を見つける
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 結果チェック（yとy_がイコールの場合True、異なる場合Falseとなるようなリストを作成）
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 正解率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TensorFlowの変数の初期化
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    print("START TRAINING")

    # 教師データ作成
    data_train, label_train, data_test, label_test = dataset.create_dataset()

    # ラベルをOneHot形式に変換
    label_train = np.eye(len(dataset.dir_names))[label_train]
    label_test = np.eye(len(dataset.dir_names))[label_test]

    # 訓練データ作成
    training_xy = [(data, label) for (data, label) in zip(data_train, label_train)]

    # トレーニング（20000回のイテレーション）
    start = time.time()
    for i in range(12001):
        # training_xyは、事前準備しておいた学習用データ（文書ベクトルとラベルがタプルで入っているリスト）
        # ミニバッチ（100件ランダムで取得）
        samples = random.sample(training_xy, 100)
        batch_xs = [s[0] for s in samples]
        batch_ys = [s[1] for s in samples]
        # 確率的勾配降下法を使い最適なパラメータを求める
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # 2000回毎に正解率を確認
        if i % 2000 == 0:
            a = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            elapsed_time = time.time() - start
            print("TRAINING(%d): %.0f%% : time=%d" % (i, (a * 100.0), elapsed_time))

    # テストデータで正解率を確認
    # test_x, test_yは事前準備しておいたテストデータ
    a = sess.run(accuracy, feed_dict={x: data_test, y_: label_test})
    print("TEST DATA ACCURACY: %.0f%%" % (a * 100.0))

    with open('bow_nn2.pickle', mode='wb') as f:
        pickle.dump(sess, f)

    pass
