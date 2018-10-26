import re
import string
import time
import pandas as pd
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm

FILENAME = "CellPhoneReview.json"


def main():
    print('Reading data...')
    df = pd.read_json(FILENAME, lines=True)
    review_group_by_ratings = df.groupby(['overall'])
    review_text_by_rating_list = [list(review_group_by_ratings.get_group(key)['reviewText']) for key, item in review_group_by_ratings]

    list_of_dicts = []
    word_pool = []
    t1 = time.time()
    j = 0
    stemmer = PorterStemmer()
    stop = stopwords.words('english') + list(string.punctuation)

    for texts in review_text_by_rating_list:
        texts_tokenized = []
        for i in tqdm(range(len(texts))):
            text = texts[i]
            text_tokenized = ' '.join([stemmer.stem(i) for i in word_tokenize(text.lower()) if i not in stop])
            texts_tokenized.append(text_tokenized)
        tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words=None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts_tokenized)
        tfidf_vocab = tfidf_vectorizer.vocabulary_
        tfidf_vocab_by_index = sorted(tfidf_vocab, key=tfidf_vocab.get)
        # print(tfidf_matrix.todense().T.shape)
        # print(tfidf_matrix.todense().T)
        tfidf_score = np.sum(tfidf_matrix.todense().T, axis=1)
        word_tfidf_dict = {}
        for i in range(len(tfidf_vocab)):
            word_tfidf_dict[tfidf_vocab_by_index[i]] = tfidf_score[i].item(0)

        list_of_dicts.append(word_tfidf_dict)
        word_pool += word_tfidf_dict.keys()
        j += 1
        if j % 100 == 0:
            print('Epoch: {}  Used time: {}'.format(j, time.time() - t1))

    print("Used time: {}".format(time.time() - t1))
    t1 = time.time()
    combined_word_score = {}
    for word in word_pool:
        tfidf_list = np.zeros(5)
        dict1, dict2, dict3, dict4, dict5 = list_of_dicts
        if word in dict1.keys():
            tfidf_list[0] = dict1[word]
        if word in dict2.keys():
            tfidf_list[1] = dict2[word]
        if word in dict3.keys():
            tfidf_list[2] = dict3[word]
        if word in dict4.keys():
            tfidf_list[3] = dict4[word]
        if word in dict5.keys():
            tfidf_list[4] = dict5[word]

        score = np.dot(tfidf_list, [-2,-1,0,1,2])
        combined_word_score[word] = score
    print("Used time: {}".format(time.time() - t1))
    # print(combined_word_score)
    sorted_wordlist_by_score = sorted(combined_word_score, key=combined_word_score.get)

    print("\nWorst list")
    for word in sorted_wordlist_by_score[:30]:
        print(word, combined_word_score[word])

    best_list = sorted_wordlist_by_score[-30:]
    best_list.reverse()
    print("\nBest list")
    for word in best_list:
        print(word, combined_word_score[word])


if __name__ == "__main__":
    main()

