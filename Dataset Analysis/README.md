# Data Analysis

Installation:
```
pip install nltk
pip install pandas
pip install matplotlib
```

Run `DatasetAnalysis.py` for results.

### Results

**Popular Products and Frequent Reviewers.** Identify the top-10 products that attract the most number of
reviews, and the top-10 reviewers who have contributed most number of reviews. List the product id/user id
with the number of reviews in a table.

**Top 10 most Popular Products**

|    | pid |reviewCount|
|---:|:--------:|:----:|
|3338|B005SUHPO6|  836 |
|1575|B0042FV2SI|  690 |
|5595|B008OHNZI0|  657 |
|6649|B009RXU59C|  634 |
|249 |B000S5Q9CA|  627 |
|5318|B008DJIIG8|  510 |
|5851|B0090YGJ4I|  448 |
|6154|B009A5204K|  434 |
|7883|B00BT7RAPG|  431 |
|342 |B0015RB39O|  424 |

**Top 10 most Frequent Reviewers**

|     |     uid|reviewCount|
|----:|:------------:|:---:|
|12192|A2NYK9KWFMJV4Y|  152|
|7773 |A22CW0ZHY3NJH8|  138|
|3039 |A1EVV74UQYVKRY|  137|
|4955 |A1ODOGXEYECQQ8|  133|
|12131|A2NOW4U7W3F7RI|  132|
|16059|A36K2N527TXXJN|  124|
|6279 |A1UQBFCERIP7VJ|  112|
|2866 |A1E1LEVQ9VQNK |  109|
|1820 |A18U49406IPPIJ|  109|
|27547|AYB4ELCS5AM8P |  107|

---
**Sentence Segmentation.** Perform sentence segmentation on the reviews and show the distribution of the
data in a plot. The x-axis is the length of a review in number of sentences, and the y-axis is the number of
reviews of each length. Discuss your findings based on the plot.

Randomly sample 5 reviews (including both short reviews and long reviews) and verify whether the
sentence segmentation function/tool detects the sentence boundaries correctly. Discuss your results.

![alt text](https://github.com/gudgud96/product-review-data-analysis/blob/dataset-analysis/src/Sentences_in_review.png "Sentences")

5 samples of the reviews and its sentence segmentation can be found [here](src/sample_tokenized_sentences.txt).

The review with the most sentences can be found [here](src/long_sentence.txt)

---
**Tokenization and Stemming.** Tokenize the reviews and show two distributions of the data, one without
stemming, and the other with stemming (you may choose the stemming algorithm implemented in any
toolkit). Again, the x-axis is the length of a review in number of words (or tokens) and the y-axis is the
number of reviews of each length. Discuss your findings based on the two plots.

List the top-20 most frequent words (excluding the stop words) before and after performing stemming.
Discuss the words that you expected to be popular given the nature of the dataset (i.e., reviews of cell phones
and accessories), and the words that you do not expect to be popular in this dataset. Stop words are the words
that are commonly used but do not carry much semantic meaning such as *a, the, of, and.* You need to list the
stop words used in your analysis in the appendix of your report.

![alt text](https://github.com/gudgud96/product-review-data-analysis/blob/dataset-analysis/src/Tokens_in_review.png "Tokens")


**Top 10 Most Frequent Words Before Stemming**

|word |count|
|-----|-----|
|phone | 174345|
|case | 144658|
|one | 85413|
|like | 71795|
|i | 67298|
|great | 65970|
|use | 60771|
|screen | 59487|
|good | 57855|
|it | 57670|
|battery | 57135|
|would | 54460|
|well | 49465|
|iphone | 47732|
|get | 46324|
|charge | 44390|
|'s | 39445|
|charger | 38170|
|really | 37971|
|product | 37683|

**Top 10 Most Frequent Words After Stemming**

|stem | count |
|-----|-------|
|phone | 189479|
|case | 163276|
|use | 116695|
|charg | 91180|
|one | 90926|
|like | 79625|
|work | 75515|
|i | 67301|
|great | 66009|
|batteri | 65076|
|get | 61102|
|screen | 61067|
|good | 58073|
|it | 57977|
|would | 54460|
|look | 51808|
|fit | 49914|
|iphon | 49900|
|well | 49476|
|time | 46971|


---
**POS Tagging.**  Randomly select 5 sentences from the dataset, and apply POS tagging. Show and discuss the
tagging results.

5 samples of the reviews and its POS Tagging can be found [here](src/random5_pos_tag.txt).

