import os
import codecs
import numpy as np
import MeCab
import pickle
import random
import time
from gensim import corpora, models
from sklearn import svm, grid_search


base_path = "../livedoor"
dir_names = ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme","movie-enter","peachy","smax","sports-watch","topic-news"]


def read_corpus():
    """
    Livedoorコーパスを読み込み、文書ごとに単語ベクトルを生成
    """
    documents = []
    categories = []
    for category, dir_name in enumerate(dir_names):
        dir_name = os.path.join(base_path, dir_name)
        for filename in os.listdir(dir_name):
            filename = os.path.join(dir_name, filename)

            # 1記事について全行読み込む
            # (1〜2行目は本文と関係ないので除去)
            text = codecs.open(filename, "r", "utf-8").readlines()[2:]  # for removing the date (1st line)

            # 全行を１つのテキストに結合する
            text = u"".join(text)

            # テキストを単語に分割
            words = tokenize(text)
            documents.append(words)
            categories.append(category)

    return documents, categories


def create_bow_vec(documents):
    """
    文章の単語リストをベクトルに変換する
    """
    dic = corpora.Dictionary(documents)

    # 単語辞書から出現頻度の少ない単語及び出現頻度の多すぎる単語を排除
    dic.filter_extremes(no_below=20, no_above=0.3)

    # Bag of Wordsベクトルの作成
    bow_corpus = [dic.doc2bow(d) for d in documents]
    return bow_corpus, dic


def tokenize(text):
    """
    MeCabを使って文章を主要な単語に分割する
    """
    words = []
    stoplist = None

    mecab = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/')

    mecab.parseToNode('')  # 空文字列をparse(node.surfaceで文字を取得できない不具合を回避できる)
    node = mecab.parseToNode(text)  # 形態素解析(分かち書き)を実施
    while node:
        # 解析結果を取得
        # featureフォーマット
        # 品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用形,活用型,原形,読み,発音
        feature = node.feature.split(',')

        # 原型の単語を取得
        word = feature[-3]
        if word == '*':
            word = node.surface

        # 主要な単語のみ出力する
        if feature[0] in ['名詞', '動詞', '形容詞'] and feature[1] != '数' and feature[-1] != 'ignore':
            if (stoplist is None) or (word not in stoplist):
                words.append(word)

        node = node.next

    return words


def create_dataset(data, categories, per_train=0.9):
    """
    学習用データセットを作成
    """
    data_train = []
    data_test = []
    label_train = []
    label_test = []

    categories_uniq = list(set(categories))
    for category in categories_uniq:
        # 対象カテゴリのデータのみ抽出
        category_fill = np.array([category] * len(data))
        categories_mask = categories == category_fill
        category_label = categories[categories_mask]
        category_data = data[categories_mask]

        # データをシャッフルする
        random.shuffle(category_data)

        # 訓練データが十分に確保できないときはエラーを出力
        num_category = len(category_label)
        num_train = num_category * per_train
        num_test = num_category - num_train
        if num_test < 0:
            raise RuntimeError("runtime error : num_test < 0")

        # 訓練データとテストデータに分割
        category_data_train = category_data[:num_train]
        category_data_test = category_data[num_train:]
        category_label_train = category_label[:num_train]
        category_label_test = category_label[num_train:]

        # データセットに追加
        data_train.extend(category_data_train)
        data_test.extend(category_data_test)
        label_train.extend(category_label_train)
        label_test.extend(category_label_test)
        print("category={0} : train={1} , test={2}".format(category, len(category_data_train), len(category_data_test)))

    return data_train, label_train, data_test, label_test


if __name__ == '__main__':
    # Livedoorコーパスから文章ごとに単語リストを抽出
    """
    documents, categories = read_corpus()
    with open('documents.pickle', mode='wb') as f:
        pickle.dump(documents, f)
    with open('categories.pickle', mode='wb') as f:
        pickle.dump(categories, f)
    """

    with open('documents.pickle', mode='rb') as f:
        documents = pickle.load(f)
    with open('categories.pickle', mode='rb') as f:
        categories = pickle.load(f)

    categories = np.array(categories)

    # 単語リストを単語ベクトルに変換
    """
    bow_corpus, dic = create_bow_vec(documents)
    with open('bow_corpus.pickle', mode='wb') as f:
        pickle.dump(bow_corpus, f)
    with open('dic.pickle', mode='wb') as f:
        pickle.dump(dic, f)
    """

    with open('bow_corpus.pickle', mode='rb') as f:
        bow_corpus = pickle.load(f)
    with open('dic.pickle', mode='rb') as f:
        dic = pickle.load(f)
    print("dic          : len={0}".format(len(dic)))
    print("bow_corpus   : len={0} , len[0]={1} , value[0]={2}".format(len(bow_corpus), len(bow_corpus[0]), bow_corpus[0]))

    # TF-IDFによる重み付け
    """
    tfidf_model = models.TfidfModel(bow_corpus)
    tfidf_corpus = np.array(tfidf_model[bow_corpus])
    with open('tfidf_corpus.pickle', mode='wb') as f:
        pickle.dump(tfidf_corpus, f)
    """

    with open('tfidf_corpus.pickle', mode='rb') as f:
        tfidf_corpus = pickle.load(f)
    print("tfidf_corpus : len={0} , len[0]={1} , value[0]={2}".format(len(tfidf_corpus), len(tfidf_corpus[0]), tfidf_corpus[0]))

    # LSIによる次元削減
    """
    lsi_model = models.LsiModel(tfidf_corpus, id2word=dic, num_topics=300)
    lsi_corpus = np.array(lsi_model[tfidf_corpus])
    lsi_corpus = lsi_corpus[:,:,1]
    with open('lsi_corpus.pickle', mode='wb') as f:
        pickle.dump(lsi_corpus, f)
    """

    with open('lsi_corpus.pickle', mode='rb') as f:
        lsi_corpus = pickle.load(f)
    print("lsi_corpus   : len={0} , len[0]={1} , value[0]={2}".format(len(lsi_corpus), len(lsi_corpus[0]), lsi_corpus[0]))

    # 教師データ作成
    data_train, label_train, data_test, label_test = create_dataset(data=lsi_corpus, categories=categories, per_train=0.9)

    # Scikit-learnのSVMアルゴリズムを使ってトレーニング
    svc = svm.SVC()
    # コストパラメータの設定（Grid Search用に複数設定）
    cs = [0.001, 0.01, 0.1, 1, 10]
    #cs = [0.1]
    # RBFカーネルパラメータの設定（Grid Search用に複数設定）
    gammas = [0.001, 0.01, 0.1, 1]
    #gammas = [0.01]
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
    # train=90% :
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time / 60) + "[min]")
    print('Grid Score: %s' % clf.grid_scores_)

    # テストデータの実行
    # Grid Searchを実行した場合、一番スコアの高いハイパーパラメータを使ったモデルでスコアを出してくれる
    # (memo)
    # train=100 : Score: 0.810376775788
    # train=200 : Score: 0.875179340029
    # train=300 : Score: 0.898631308811
    # train=90% :
    start = time.time()
    print("check test data...")
    score = clf.score(data_test, label_test)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time / 60) + "[min]")
    print('Test Score: ' + str(score))

    pass