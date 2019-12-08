from pickle import dump, load
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

names = load(open('data/names.txt', 'rb'))
print('Names length: ', len(names))
classic_train_vector = sparse.load_npz('classic/train_vector.npz')
classic_test_vector = sparse.load_npz('classic/test_vector.npz')
print('Classic vectors length: ', len(
    classic_train_vector) + len(classic_test_vector))

bert_train_vector = np.load('bert/train_vector.npy')
bert_test_vector = np.load('bert/test_vector.npy')

print('Bert vectors length: ', len(
    classic_train_vector) + len(classic_test_vector))

print('Fitting classic model...')
classic_nn = NearestNeighbors(n_neighbors=5)
classic_nn.fit(classic_train_vector)

dump(classic_nn, open('classic/knn.sav', 'wb'))

print('Fitting bert model...')
bert_nn = NearestNeighbors(n_neighbors=5)
bert_nn.fit(bert_train_vector)

dump(bert_nn, open('bert/knn.sav', 'wb'))


def buildArticleRecommender(model, article_title, vectorized_plots):
    pred = model.kneighbors(vectorized_plots)

    def recommend(query):
        try:
            idx = pred[1][article_title.index(query)]
            print(idx)
            for i in idx:
                print(article_title[i])
        except ValueError:
            print("{} not found in article database. Suggestions:")
            for i, name in enumerate(article_title):
                if query.lower() in name.lower():
                    print(i, name)
    return recommend


bertRecommend = buildArticleRecommender(bert_nn, names, bert_test_vector)
classicRecommend = buildArticleRecommender(
    classic_nn, names, classic_train_vector)

print(names[6])
print(bertRecommend(names[6]))
print(classicRecommend(names[6]))
