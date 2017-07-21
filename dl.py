import os
import numpy as np
import codecs
import pickle
from gensim.models import word2vec
import dataset


def read_corpus(doc_path, dirs):
    """
    Livedoorコーパスを読み込み、文書ごとに単語ベクトルを生成
    """
    documents = []
    categories = []
    for category, dir_name in enumerate(dirs):
        dir_name = os.path.join(doc_path, dir_name)
        for filename in os.listdir(dir_name):
            filename = os.path.join(dir_name, filename)

            # 1記事について全行読み込む
            # (1〜2行目は本文と関係ないので除去)
            text = codecs.open(filename, "r", "utf-8").readlines()[2:]  # for removing the date (1st line)

            # 全行を１つのテキストに結合する
            text = u"".join(text)

            # テキストを単語に分割
            words = dataset.tokenize(text)
            documents.append(words)
            categories.append(category)

    return documents, categories


if __name__ == '__main__':
    #read_corpus(dataset.base_path, dataset.dir_names)
    model = word2vec.Word2Vec.load("../jawiki/model/data.w2v")
    with open('w2v.pickle', mode='wb') as f:
        pickle.dump(model, f)
    with open('w2v.pickle', mode='rb') as f:
        model = pickle.load(f)

    print(model["山"])

    with open('documents.pickle', mode='rb') as f:
        documents = pickle.load(f)
    with open('categories.pickle', mode='rb') as f:
        categories = pickle.load(f)



    categories = np.array(categories)


    pass
