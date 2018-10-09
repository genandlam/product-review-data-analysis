from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import json
import numpy as np


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
    FILENAME = "CellPhoneReview-20000.json"
    print('Reading data...')
    review_data = open(FILENAME).readlines()
    documents = [json.loads(d)['reviewText'] for d in review_data]

    print("Processing...")
    no_features = 1000

    # NMF is able to use tf-idf
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # print(tfidf_feature_names)

    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    no_topics = 5

    # Run NMF
    nmf = NMF(n_components=no_topics, random_state=10, alpha=.1, l1_ratio=.5, init='nndsvda').fit(tfidf)

    # Run LDA
    # lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

    no_top_words = 6
    nmf_topics = get_topics(nmf, tfidf_feature_names, no_top_words)
    # print('\n')
    # display_topics(lda, tf_feature_names, no_top_words)

    w = nmf.fit_transform(tfidf)            # topic of each document
    h = nmf.components_                     # rank for each word for a topic
    print("{} {}".format(w.shape,h.shape))
    doc_topics = np.argmax(w, axis=1)

    with open('nmf_topic_modelling.txt', 'w+') as txt:
        for i in range(len(documents)):
            txt.write(documents[i] + '\n')
            txt.write('Topic: ' + str(doc_topics[i]) + ' ' + ' '.join(nmf_topics[doc_topics[i]]) + '\n')
            txt.write('\n')


if __name__ == "__main__":
    main()