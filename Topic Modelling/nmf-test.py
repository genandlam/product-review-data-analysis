import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm


def get_topics(model, feature_names, no_top_words):
    topic_list = []
    for topic_idx, topic in enumerate(model.components_):
        topic_keywords = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        print("Topic %d:" % (topic_idx))
        print(" ".join(topic_keywords))
        topic_list.append(topic_keywords)
    return topic_list


def predict(doc, model):
    return model.transform(doc)


def main():
    t = time.time()
    is_nmf = False
    FILENAME = "CellPhoneReview.json"
    print('Reading data...')
    review_data = open(FILENAME).readlines()
    documents = [json.loads(d)['reviewText'] for d in tqdm(review_data)]
    ratings = [json.loads(d)['overall'] for d in tqdm(review_data)]

    print("Processing...")
    no_features = 1000

    # NMF is able to use tf-idf
    if is_nmf:
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(tqdm(documents))
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        print(tfidf_feature_names)

    else:
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 5

    # Run NMF
    if is_nmf:
        nmf = NMF(n_components=no_topics, random_state=10, alpha=.1, l1_ratio=.5, init='nndsvda', verbose=True).fit(tfidf)

    # Run LDA
    else:
        lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=0, verbose=1).fit(tf)

    no_top_words = 6
    if is_nmf:
        nmf_topics = get_topics(nmf, tfidf_feature_names, no_top_words)
        w = nmf.transform(tfidf)
        h = nmf.components_
        np.save('topic_proba_nmf.npy', w)

    else:
        lda_topics = get_topics(lda, tf_feature_names, no_top_words)
        w = lda.transform(tf)            # topic of each document
        h = lda.components_                     # rank for each word for a topic
        np.save('topic_proba_lda.npy', w)

    print("{} {}".format(w.shape,h.shape))
    doc_topics = np.argmax(w, axis=1)
    ranking_counts = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]

    if is_nmf:
        txt = open('nmf_topic_modelling.txt', 'w+')
    else:
        txt = open('lda_topic_modelling.txt', 'w+')

    for i in range(len(documents)):
        txt.write(documents[i] + '\n')
        if is_nmf:
            txt.write('Topic: ' + str(doc_topics[i]) + ' ' + ' '.join(nmf_topics[doc_topics[i]]) + '\n')
        else:
            txt.write('Topic: ' + str(doc_topics[i]) + ' ' + ' '.join(lda_topics[doc_topics[i]]) + '\n')
        txt.write('Ratings: ' + str(ratings[i]) + '\n')
        txt.write('\n')
        ranking_counts[doc_topics[i]][int(ratings[i]) - 1] += 1

    average_ranking = []
    for i in range(len(ranking_counts)):
        weighted_sum = sum([ranking_counts[i][j] * (j + 1) for j in range(len(ranking_counts[i]))])
        average = weighted_sum / sum(ranking_counts[i])
        average_ranking.append(average)

    print(ranking_counts)
    print(average_ranking)

    print("Total time used: {}".format(time.time() - t))


if __name__ == "__main__":
    main()