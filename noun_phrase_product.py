import nltk
import json
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import os
from IPython.display import Image, display
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from textblob import TextBlob
from textblob.en.taggers import PatternTagger
from textblob.np_extractors import ConllExtractor
import csv
from collections import Counter

data = [json.loads(line) for line in open('CellPhoneReview.json')]

print ("product 1 = B005SUHPO6")
print("product 2 = B0042FV2SI")
print("product 3 = B008OHNZI0")

product_review1=[]
product_review2=[]
product_review3=[]
#phone_review_texts=[]
charger_review_texts=[]
for item_dic in data:
#    print(item_dic)
    pid=item_dic.get('asin')
    if pid== 'B005SUHPO6':
        reviewText=item_dic.get('reviewText')
        product_review1.append(reviewText)
    if pid== 'B0042FV2SI':
        reviewText=item_dic.get('reviewText')
        product_review2.append(reviewText)
    if pid == 'B008OHNZI0':
        reviewText=item_dic.get('reviewText')
        product_review3.append(reviewText)

grammar = ('''
  
    NPs:{<RB.?>*<VB.?>*<NNP>+} 
   
    NP: {<JJ>*<NN>} # NP
    
    ''')
chunkParser = nltk.RegexpParser(grammar)

better_tree=[] # PatternTagger 
products[product_review3,product_review2,product_review1]

for counter,product in enumerate(products):
	better_tree=[]
	for texts in product_review3:
	    tokens = nltk.word_tokenize(texts)
	    words = [word for word in tokens if word.isalpha()]
	#     print(tagged_word)
	    pattern_tagger = PatternTagger()
	    extractor = ConllExtractor()
	    tagged_word_2 = TextBlob(' '.join(words), pos_tagger=pattern_tagger, np_extractor=extractor)
	#     print(tagged_word_2.tags)
	    # PatternTagger 
	    better_tree.append(chunkParser.parse(tagged_word_2.tags))

	np_trees=[]

	for tree in better_tree:
	    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'or t.label() == 'NPs' ):
	#        print(subtree)
	        if len(subtree)>1:
	            np_trees.append(subtree)
	       #     print(len(subtree))
	noun_phrases=[]    

	for np in np_trees:
	    noun_phrases.append(' '.join([w for w, t in np.leaves()]))
	    

	esBigramFreq = Counter(noun_phrases)
	esBigramFreq.most_common(10)
	print('Top 10 noun phrases product {}:'.format(counter) )
	print(esBigramFreq.most_common(10))