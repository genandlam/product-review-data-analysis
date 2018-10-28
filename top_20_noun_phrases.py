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

review_texts=[]
for item_dic in data:
#    print(item_dic)
    reviewText=item_dic.get('reviewText')
    review_texts.append(reviewText)
    
grammar = ('''
  
    NPs:{<RB.?>*<VB.?>*<NNP>+} 
   
    NP: {<JJ>*<NN>} # NP
    
    ''')

chunkParser = nltk.RegexpParser(grammar)

trees1=[] #nltk tagger
better_tree=[] # PatternTagger 
for texts in review_texts:
    tokens = nltk.word_tokenize(texts)
    words = [word for word in tokens if word.isalpha()]
    tagged_word=nltk.pos_tag(words)
#     print(tagged_word)
    pattern_tagger = PatternTagger()
    extractor = ConllExtractor()
    tagged_word_2 = TextBlob(' '.join(words), pos_tagger=pattern_tagger, np_extractor=extractor)
#     print(tagged_word_2.tags)
    trees1.append(chunkParser.parse(tagged_word))
    # PatternTagger 
    better_tree.append(chunkParser.parse(tagged_word_2.tags))

    
np_trees=[]
np_trees_str=[]
for tree in better_tree:
    
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'or t.label() == 'NPs' ):
        if len(subtree)>1:
            np_trees.append(subtree)
            print(len(subtree))
        tr1 = str(subtree)
        np_trees_str.append(tr1)
        
noun_phrases=[]    
for np in np_trees:
    noun_phrases.append(' '.join([w for w, t in np.leaves()]))
    
    
esBigramFreq = Counter(noun_phrases)
esBigramFreq.most_common(20)
print(esBigramFreq.most_common(20))





