import numpy as np
import pickle
from gensim import models
import dataset

if __name__ == '__main__':
    with open('documents.pickle', mode='rb') as f:
        documents = pickle.load(f)

    documents_test, categories = dataset.read_corpus(
        "test",
        ["dokujo-tsushin"])
    test_bow_corpus, dic = dataset.create_bow_vec(documents_test, no_below=0, no_above=1.0)
    with open('bow_corpus.pickle', mode='rb') as f:
        bow_corpus = pickle.load(f)
    with open('dic.pickle', mode='rb') as f:
        dic = pickle.load(f)

    test_bow_corpus.extend(bow_corpus)

    tfidf_model = models.TfidfModel(test_bow_corpus)
    tfidf_corpus = np.array(tfidf_model[test_bow_corpus])
    lsi_model = models.LsiModel(tfidf_corpus, id2word=dic, num_topics=300)
    lsi_corpus = np.array(lsi_model[tfidf_corpus])
    lsi_corpus = lsi_corpus[:, :, 1]

    with open('bow_svm.pickle', mode='rb') as f:
        clf = pickle.load(f)

    test_lsi_corpus = lsi_corpus[0:4]

    predict = clf.predict(test_lsi_corpus)
    print(predict)
