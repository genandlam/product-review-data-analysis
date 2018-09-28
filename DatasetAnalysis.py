import time
import nltk
import heapq
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# functions are defined here
def save_results_to_file(save, dataframe, filename):
    if (save):
        print("RESULTS SAVED TO FILE")
        dataframe.to_csv(filename, header=True, index=False, sep='\t')
    return


def get_stem(word_list):
    stem_list = []
    for word in word_list:
        stem_list.append(ps.stem(word))
    return stem_list


def sentence_tokenize(row):
    sent_tokens = nltk.sent_tokenize(row['reviewText'])
    row['sentsToken'] = sent_tokens
    row['numSentsToken'] = len(sent_tokens)
    return row


# tokenize words and then remove punctuations
def word_tokenize(row):
    word_tokens = nltk.word_tokenize(row['reviewText'])
    word_tokens = [i for i in word_tokens if i not in string.punctuation]
    row['wordToken'] = word_tokens
    row['numWordToken'] = len(word_tokens)
    return row


def word_tokenize_stemming(row):
    word_tokens = nltk.word_tokenize(row['reviewText'])
    word_tokens = [i for i in word_tokens if i not in string.punctuation]
    word_tokens_stem = get_stem(word_tokens)
    row['wordTokenStem'] = word_tokens_stem
    row['numWordTokenStem'] = len(word_tokens_stem)
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
    row['posTag'] = nltk.pos_tag(row['wordToken'])
    return row


# set display options
pd.options.display.max_colwidth = 500

# define constants
SAVE_RESULTS_TO_FILE = False
JSON_FILE_PATH = "CellPhoneReview\CellPhoneReview.json"

data = pd.read_json(JSON_FILE_PATH, lines=True)

# data = data.head(20)
print("JSON DATA READ")


# top-10 products that attract the most number of reviews

reviews_per_product = data.groupby(['asin'])['reviewText'].count()
reviews_per_product = pd.DataFrame({'pid':reviews_per_product.index, 'reviewCount':reviews_per_product.values})

reviews_per_product.sort_values(by=['reviewCount'], ascending=False, inplace=True)
print(reviews_per_product.head(10))

save_results_to_file(SAVE_RESULTS_TO_FILE, reviews_per_product.head(10), "top10_pid_reviews.txt")


# top-10 reviewers who have contributed most number of reviews

reviews_per_user = data.groupby(['reviewerID'])['reviewText'].count()
reviews_per_user = pd.DataFrame({'uid':reviews_per_user.index, 'reviewCount':reviews_per_user.values})

reviews_per_user.sort_values(by=['reviewCount'], ascending=False, inplace=True)
print(reviews_per_user.head(10))

save_results_to_file(SAVE_RESULTS_TO_FILE, reviews_per_user.head(10), "top10_uid_reviews.txt")


# perform sentence segmentation on the reviews and show the distribution of the
# data in a plot. The x-axis is the length of a review in number of sentences,
# and the y-axis is the number of reviews of each length.

start_time = time.time()

tokenized_sentences = data.apply(sentence_tokenize, axis=1)

print(time.time() - start_time)

save_results_to_file(SAVE_RESULTS_TO_FILE, tokenized_sentences, "tokenized_sentences.txt")

numSents_per_review = tokenized_sentences.groupby(['numSentsToken'])['reviewText'].count()

plt.plot(numSents_per_review)
plt.xlabel('Sentences in a Review')
plt.ylabel('Reviews')
plt.show()


# tokenize the reviews and show two distributions of the data, one without
# stemming, and the other with stemming. Again, the x-axis is the length of a
# review in number of words (or tokens) and the y-axis is the number of reviews
# of each length.

start_time = time.time()

tokenized_words = data.apply(word_tokenize, axis=1)

print(time.time() - start_time)

save_results_to_file(SAVE_RESULTS_TO_FILE, tokenized_words, "tokenized_words.txt")

numWords_per_review = tokenized_words.groupby(['numWordToken'])['reviewText'].count()

plt.subplot(2,1,1)
plt.plot(numWords_per_review)
plt.xlabel('Tokens in a Review')
plt.ylabel('Reviews')


# tokenize reviews with stemming

ps = nltk.PorterStemmer()

start_time = time.time()

tokenized_words_stem = data.apply(word_tokenize_stemming, axis=1)

print(time.time() - start_time)

save_results_to_file(SAVE_RESULTS_TO_FILE, tokenized_words_stem, "tokenized_words_stem.txt")

numWordsStem_per_review = tokenized_words_stem.groupby(['numWordTokenStem'])['reviewText'].count()

plt.subplot(2,1,2)
plt.plot(numWordsStem_per_review)
plt.xlabel('Token Stems in a Review')
plt.ylabel('Reviews')

plt.show()


# list the top 20 most frequent words before stemming
stop_words = ['a', 'the', 'I', 'it', 'to', 'my', 'and', 'for', 'is', 'of' ,'they', 'in', 'it']

unigram_counts = get_unigram_counts(tokenized_words['wordToken'].values)

top_n_unigramCounts = heapq.nlargest(10+len(stop_words), unigram_counts, key=unigram_counts.get)

print("TOP 10 MOST FREQUENT WORDS BEFORE STEMMING")
for stem in top_n_unigramCounts:
    if stem not in stop_words:
        print(stem + " : "+ str(unigram_counts[stem]))



# list the top 20 most frequent words after stemming
unigram_counts = get_unigram_counts(tokenized_words_stem['wordTokenStem'].values)

top_n_unigramCounts = heapq.nlargest(10+len(stop_words), unigram_counts, key=unigram_counts.get)

print("TOP 10 MOST FREQUENT WORDS AFTER STEMMING")
for stem in top_n_unigramCounts:
    if stem not in stop_words:
        print(stem + " : "+ str(unigram_counts[stem]))


# randomly select 5 sentences from the dataset, and apply POS tagging

sample_data = data.ix[np.random.choice(data.index, 5)]

tokenized_words_sample = sample_data.apply(word_tokenize, axis=1)
posTag_sample = tokenized_words_sample.apply(pos_tag, axis=1)

save_results_to_file(SAVE_RESULTS_TO_FILE, posTag_sample, "random5_pos_tag.txt")
print(posTag_sample.posTag)