import os
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
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


def main():
    t = time.time()
    FILENAME = "CellPhoneReview.json"
    print('Reading data...')
    review_data = open(FILENAME).readlines()
    documents = [json.loads(d)['reviewText'] for d in tqdm(review_data)]
    ratings = [json.loads(d)['overall'] for d in tqdm(review_data)]

    if os.path.isfile('topic_proba_lda.npy'):
        w = np.load('topic_proba_lda.npy')

    else:
        print("Processing...")
        no_features = 1000

        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()

        no_topics = 5

        # Run LDA
        lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50., random_state=0, verbose=1).fit(tf)

        no_top_words = 6

        lda_topics = get_topics(lda, tf_feature_names, no_top_words)
        w = lda.transform(tf)            # topic of each document
        h = lda.components_              # rank for each word for a topic
        np.save('topic_proba_lda.npy', w)

    print("{}".format(w.shape))
    doc_topics = np.argmax(w, axis=1)
    ranking_counts = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]

    # Evaluation
    f1_score(documents, doc_topics)

    txt = open('lda_topic_modelling.txt', 'w+')

    for i in range(len(documents)):
        txt.write(documents[i] + '\n')

        # I find topic 3 meaningless, so I assign a second highest topic for it
        if doc_topics[i] == 3:
            doc_topics[i] = np.argsort(w[i])[-2]
            txt.write('Topic: ' + str(doc_topics[i]) + ' ' + '\n')

        txt.write('Ratings: ' + str(ratings[i]) + '\n')
        txt.write('Topic proba: {}'.format(str(w[i])))
        txt.write('\n')
        ranking_counts[doc_topics[i]][int(ratings[i]) - 1] += 1

    average_ranking = []
    for i in range(len(ranking_counts)):
        weighted_sum = sum([ranking_counts[i][j] * (j + 1) for j in range(len(ranking_counts[i]))])
        average = 0 if sum(ranking_counts[i]) == 0 else weighted_sum / sum(ranking_counts[i])
        average_ranking.append(average)

    print(ranking_counts)
    print(average_ranking)
    print("Total time used: {}".format(time.time() - t))


def f1_score(document, doc_topics):
    indices = np.random.choice(len(document), 20)
    print(indices)
    with open('f1_score.csv', 'w+') as f:
        for index in indices:
            f.write("{},{}\n".format(document[index].replace(',', ';'), doc_topics[index]))


if __name__ == "__main__":
    main()