import os
import codecs
import MeCab
import pickle
from gensim import corpora
from gensim import models


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

    # TF-IDFによる重み付け
    """
    tfidf_model = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]
    with open('tfidf_corpus.pickle', mode='wb') as f:
        pickle.dump(tfidf_corpus, f)
    """

    with open('tfidf_corpus.pickle', mode='rb') as f:
        tfidf_corpus = pickle.load(f)

    # LSIによる次元削減
    """
    lsi_model = models.LsiModel(tfidf_corpus, id2word=dic, num_topics=300)
    lsi_corpus = lsi_model[tfidf_corpus]
    with open('lsi_corpus.pickle', mode='wb') as f:
        pickle.dump(lsi_corpus, f)
    """

    with open('lsi_corpus.pickle', mode='rb') as f:
        lsi_corpus = pickle.load(f)

    pass
