import time
import nltk
import heapq
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# functions are defined here
def save_to_file(save, dataframe, filename):
    if save:
        print("RESULTS SAVED TO: " + filename)
        dataframe.to_csv(filename, header=True, index=False, sep='\t')
    return


def remove_stopwords(row):
    row = [w for w in row.split() if w not in set(stopwords)]  # remove stopwords
    return ' '.join(row)


def get_stem(word_list):
    stem_list = []
    for word in word_list:
        stem_list.append(stemmer.stem(word))
    return stem_list


def sentence_tokenize(row):
    sent_tokens = nltk.sent_tokenize(row['reviewText'])
    row['sentences'] = sent_tokens
    row['numSentences'] = len(sent_tokens)
    return row


# tokenize words and then remove punctuations
def word_tokenize(row):
    word_tokens = nltk.word_tokenize(row['reviewText'])
    word_tokens = [i for i in word_tokens if i not in string.punctuation]
    word_tokens_stem = get_stem(word_tokens)
    row['words'] = word_tokens
    row['numWords'] = len(word_tokens)
    row['wordStems'] = word_tokens_stem
    row['numWordStems'] = len(word_tokens_stem)
    return row


def get_unigram_counts(stems_list):
    unigrams = {}
    for idx in range(stems_list.shape[0]):
        for stem in stems_list[idx]:
            if stem not in unigrams:
                unigrams[stem] = 1
            else:
                unigrams[stem] += 1
    return unigrams


def pos_tag(row):
    row['posTag'] = nltk.pos_tag(row['words'])
    return row


# set display options
pd.options.display.max_colwidth = 500

# define constants
SAVE_TO_FILE = True
JSON_FILE_PATH = "CellPhoneReview\CellPhoneReview.json"

stemmer = nltk.PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english')).union(list(string.punctuation))
data_original = pd.read_json(JSON_FILE_PATH, lines=True)

# perform some preprocessing on the data
data = data_original.drop(['overall', 'reviewTime', 'summary', 'unixReviewTime'], axis=1)
data['reviewText'] = data['reviewText'].astype(str).str.lower()
# data = data[:1000]

# ----------------------------------- Popular Products and Frequent Reviewers ---------------------------------------- #
# top-10 products that attract the most number of reviews

reviews_per_product = data.groupby(['asin'])['reviewText'].count()
reviews_per_product = pd.DataFrame({'pid': reviews_per_product.index, 'reviewCount': reviews_per_product.values})
reviews_per_product.sort_values(by=['reviewCount'], ascending=False, inplace=True)
print(reviews_per_product.head(10))
save_to_file(SAVE_TO_FILE, reviews_per_product.head(10), "top10_pid_reviews.txt")


# top-10 reviewers who have contributed most number of reviews

reviews_per_user = data.groupby(['reviewerID'])['reviewText'].count()
reviews_per_user = pd.DataFrame({'uid': reviews_per_user.index, 'reviewCount': reviews_per_user.values})
reviews_per_user.sort_values(by=['reviewCount'], ascending=False, inplace=True)
print(reviews_per_user.head(10))
save_to_file(SAVE_TO_FILE, reviews_per_user.head(10), "top10_uid_reviews.txt")

# --------------------------------------------- Sentence Segmentation ------------------------------------------------ #
# perform sentence segmentation on the reviews and show the distribution of the
# data in a plot. The x-axis is the length of a review in number of sentences,
# and the y-axis is the number of reviews of each length.

token_sentences = data.apply(sentence_tokenize, axis=1)
save_to_file(SAVE_TO_FILE, token_sentences, "tokenized_sentences.txt")

sentences_per_review = token_sentences.groupby(['numSentences'])['reviewText'].count()

plt.plot(sentences_per_review)
plt.xlabel('Number of Sentences')
plt.ylabel('Reviews')
plt.show()


# randomly sample 5 sentences from the dataset
sample_token_sentences = token_sentences.ix[np.random.choice(data.index, 5)]
print(sample_token_sentences['reviewText'])
save_to_file(SAVE_TO_FILE, sample_token_sentences, "sample_tokenized_sentences.txt")

# find the outlier
save_to_file(SAVE_TO_FILE, token_sentences.loc[token_sentences['numSentences'].idxmax()], "long_sentence.txt")


# -------------------------------------------- Tokenization and Stemming --------------------------------------------- #
# tokenize the reviews and show two distributions of the data, one without
# stemming, and the other with stemming. Again, the x-axis is the length of a
# review in number of words (or tokens) and the y-axis is the number of reviews
# of each length.

stemmer = nltk.PorterStemmer()
tokenized_data = data.apply(word_tokenize, axis=1)

token_words = tokenized_data[['reviewText', 'numWords', 'words']]
save_to_file(SAVE_TO_FILE, token_words[['reviewText', 'words']], "token_words.txt")

words_per_review = token_words.groupby(['numWords'])['reviewText'].count()

plt.subplot(2, 1, 1)
plt.plot(words_per_review)
plt.xlabel('Number of Words')
plt.ylabel('Reviews')

# randomly sample 5 sentences from the dataset
sample_token_words = token_words.ix[np.random.choice(data.index, 5)]
save_to_file(SAVE_TO_FILE, sample_token_words, "sample_token_words.txt")


# tokenize reviews with stemming
token_words_stemmed = tokenized_data[['reviewText', 'numWordStems', 'wordStems']]
save_to_file(SAVE_TO_FILE, token_words_stemmed[['reviewText', 'numWordStems', 'wordStems']], "token_words_stemmed.txt")

# randomly sample 5 sentences from the dataset
sample_token_words_stemmed = token_words_stemmed.ix[np.random.choice(data.index, 5)]
save_to_file(SAVE_TO_FILE, sample_token_words_stemmed, "sample_token_words_stem.txt")

word_stems_per_review = token_words_stemmed.groupby(['numWordStems'])['reviewText'].count()

plt.subplot(2, 1, 2)
plt.plot(word_stems_per_review)
plt.xlabel('Number of Word Stems')
plt.ylabel('Reviews')
plt.show()


data['reviewText'] = data['reviewText'].apply(remove_stopwords)
tokenized_data = data.apply(word_tokenize, axis=1)

# list the top 20 most frequent words before stemming
unigram_counts = get_unigram_counts(tokenized_data['words'].values)
top_n_unigrams = heapq.nlargest(20, unigram_counts, key=unigram_counts.get)

print("TOP 20 MOST FREQUENT WORDS BEFORE STEMMING")
for word in top_n_unigrams:
    print(word + " : " + str(unigram_counts.get(word)))


# list the top 20 most frequent words after stemming
unigram_counts = get_unigram_counts(tokenized_data['wordStems'].values)
top_n_unigrams = heapq.nlargest(20, unigram_counts, key=unigram_counts.get)

print("TOP 20 MOST FREQUENT WORDS AFTER STEMMING")
for word in top_n_unigrams:
    print(word + " : " + str(unigram_counts.get(word)))

# --------------------------------------------------- POS Tagging ---------------------------------------------------- #
# randomly select 5 sentences from the dataset, and apply POS tagging
sample_data = data.ix[np.random.choice(data.index, 5)]

tokenized_words_sample = sample_data.apply(word_tokenize, axis=1)
posTag_sample = tokenized_words_sample.apply(pos_tag, axis=1)

save_to_file(SAVE_TO_FILE, posTag_sample, "random5_pos_tag.txt")
print(posTag_sample.posTag)
