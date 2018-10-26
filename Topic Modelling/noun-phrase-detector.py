import json
from textblob import TextBlob
from textblob.en.taggers import NLTKTagger, PatternTagger
from textblob.np_extractors import ConllExtractor
import nltk

'''
 ( ) - Remove phrases with only 1 word because it is meaningless
 ( ) - Revise the regex pattern
'''
def main():
    # FILENAME = "CellPhoneReview-1000.json"
    # print('Reading data...')
    # review_data = open(FILENAME).readlines()
    # document = [json.loads(d)['reviewText'] for d in review_data][0]
    document = "These are awesome and make my phone look so stylish! I have only used one so far and have had it on for almost a year! CAN YOU BELIEVE THAT! ONE YEAR!! Great quality!"
    print(document)
    nltk_tagger = NLTKTagger()
    extractor = ConllExtractor()
    blob = TextBlob(document, pos_tagger=nltk_tagger, np_extractor=extractor)
    print(blob.tags)
    print(blob.noun_phrases)

    pattern_tagger = PatternTagger()
    blob2 = TextBlob(document, pos_tagger=pattern_tagger, np_extractor=extractor)
    print(blob2.tags)
    print(blob2.noun_phrases)

    tagged = nltk.pos_tag(tokenize(document.lower()))
    print(tagged)
    grammar = ('''
            NP: {<DT>?(<RB.?>*<VB.?>*<NNPS?>+<NNS?>+ | <JJ>*<NNS?>+)} # NP
            ''')

    chunkParser = nltk.RegexpParser(grammar)
    tree = chunkParser.parse(tagged)
    noun_phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            noun_phrase = ' '.join([elem[0] for elem in subtree])
            noun_phrases.append(noun_phrase)

    print(noun_phrases)
    # ratings = [json.loads(d)['overall'] for d in review_data]


def tokenize(text, include_punc=False):
    '''Return a list of word tokens.

    :param text: string of text.
    :param include_punc: (optional) whether to include punctuation as separate tokens. Default to True.
    '''
    tokens = nltk.tokenize.word_tokenize(text)
    if include_punc:
        return tokens
    else:
        # Return each word token
        # Strips punctuation unless the word comes from a contraction
        # e.g. "Let's" => ["Let", "'s"]
        # e.g. "Can't" => ["Ca", "n't"]
        # e.g. "home." => ['home']
        return [word if word.startswith("'") else strip_punc(word)
                for word in tokens if strip_punc(word)]


def strip_punc(word):
    punctuation_list = "!@#$%^&*()_+-=,.?~/"
    return ''.join([char for char in word if char not in punctuation_list])



if __name__ == "__main__":
    main()