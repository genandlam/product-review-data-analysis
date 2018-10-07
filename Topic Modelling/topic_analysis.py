import json
import time

import spacy
# spacy.load('en')
from spacy.lang.en import English
parser = English()

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import random


# nltk.download('wordnet')
# nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens



def main():
    FILENAME = "CellPhoneReview-20000.json"

    print('Reading data...')
    review_data = open(FILENAME).readlines()
    review_text = [json.loads(d)['reviewText'] for d in review_data]
    # print(review_text)

    print('Processing...')
    text_data = []
    for line in review_text:
        tokens = prepare_text_for_lda(line)
        # print(tokens)
        text_data.append(tokens)

    from gensim import corpora
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    import pickle
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    import gensim
    NUM_TOPICS = 3
    t1 = time.time()
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    t2 = time.time() - t1
    print("Training time: {}".format(t2))
    ldamodel.save('model.gensim')
    topics = ldamodel.print_topics(num_words=6)
    for topic in topics:
        print(topic)

    for text in review_text:
        new_text = prepare_text_for_lda(text)
        new_text_bow = dictionary.doc2bow(new_text)
        # print(new_text_bow)
        text_topics = ldamodel.get_document_topics(new_text_bow)
        # print(text_topics)
        topic_probs = [text_topics[i][1] for i in range(len(text_topics))]
        index = topic_probs.index(max(topic_probs))
        # print("text: {} \n topic: {} \n".format(text, topics[index]))

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model.gensim')
    import pyLDAvis.gensim
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.show(lda_display)


if __name__ == "__main__":
    main()