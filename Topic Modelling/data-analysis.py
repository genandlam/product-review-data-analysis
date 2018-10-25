import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import random
from nltk.stem.porter import *

FILENAME = "CellPhoneReview.json"


def main():
    print('Reading data...')
    df = pd.read_json(FILENAME, lines=True)
    print(df['asin'].value_counts()[:10])
    print(df['reviewerID'].value_counts()[:10])

    text_df = [sent_tokenize(txt) for txt in df['reviewText']]
    len_text_df = [len(txt) for txt in text_df]
    length_distribution = pd.Series(len_text_df).value_counts()
    dist_list = []
    for i in range(1, max(length_distribution.keys()) + 1):
        if i in length_distribution.keys():
            dist_list.append(length_distribution[i])
        else:
            dist_list.append(0)

    plt.plot(range(1, len(dist_list) + 1), dist_list)
    plt.savefig('sentence-segment.png')
    plt.close()

    random_sent = random.sample(text_df, 5)
    for sent in random_sent:
        print(sent)

    stemmer = PorterStemmer()
    stop = stopwords.words('english') + list(string.punctuation)
    punc = list(string.punctuation)

    token_df_nostop = [[i for i in word_tokenize(txt.lower()) if i not in stop] for txt in df['reviewText']]
    token_df_nostop_stem = [[stemmer.stem(i) for i in word_tokenize(txt.lower()) if i not in stop] for txt in df['reviewText']]
    token_df_stop = [[i for i in word_tokenize(txt.lower()) if i not in punc] for txt in df['reviewText']]
    token_df_stop_stem = [[stemmer.stem(i) for i in word_tokenize(txt.lower()) if i not in punc] for txt in df['reviewText']]

    def show_distribution(token_list, f_name):
        len_token = [len(token) for token in token_list]
        token_length_distribution = pd.Series(len_token).value_counts()
        dist_list = []
        for i in range(1, max(token_length_distribution.keys()) + 1):
            if i in token_length_distribution.keys():
                dist_list.append(token_length_distribution[i])
            else:
                dist_list.append(0)

        plt.plot(range(1, len(dist_list) + 1), dist_list)
        # plt.scatter(range(1, len(dist_list) + 1), dist_list)
        plt.savefig(f_name)
        plt.close()

    show_distribution(token_df_nostop, "token-nostop.png")
    show_distribution(token_df_nostop_stem, "token-nostop-stem.png")
    show_distribution(token_df_stop, "token-stop.png")
    show_distribution(token_df_stop_stem, "token-stop-stem.png")

    def flatten(lst):
        result_lst = []
        for elem in lst:
            if isinstance(elem, list):
                for e in elem:
                    result_lst.append(e)
            else:
                result_lst.append(elem)
        return result_lst

    no_stem = pd.Series(flatten(token_df_nostop)).value_counts()
    stem = pd.Series(flatten(token_df_nostop_stem)).value_counts()

    print(no_stem[:20])
    print(stem[:20])

    random_pos = random.sample(token_df_stop, 5)
    for sentence in random_pos:
        print(nltk.pos_tag(sentence))


if __name__ == "__main__":
    main()