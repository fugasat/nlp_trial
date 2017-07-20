import os
import codecs
import MeCab

base_path = "../livedoor"
dir_names = ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme","movie-enter","peachy","smax","sports-watch","topic-news"]

documents = []

def read_corpus():
    """
    Livedoorコーパスを読み込み、文書ごとに単語ベクトルを生成
    """
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

            pass


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
    read_corpus()