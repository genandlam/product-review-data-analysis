How to run sentiment_analysis:

the program is best run on command line

libraries used : 
- nltk
- numpy
- tqdm

I. INSTALLING LIBRARIES
Create a virtual environment.
- on cmd, navigate to your desired directory
- TYPE py -m pip install virtualenv  #'py' calls python executable, may be 'python' depending on installation
- TYPE py -m virtualenv NLP          # NLP can be named arbitrarily
- TYPE NLP\Scripts\Activate
- TYPE pip install nltk
- TYPE pip install numpy
- TYPE pip install tqdm

Installing nltk resources
from the same cmd instance

- TYPE py   #'py' or 'python'
- TYPE import nltk
- TYPE nltk.download()

from the nltk downloader, install
1. StopWords Corpus		#under Corpora
2. Averaged Perceptron Tagger	#under Models

-TYPE exit()

II. RUNNING PROGRAM
-on cmd, activate virtual environment created prior
-navigate to the folder with sentiment_detector.py

# to collect tokens and data, 
- create a folder named 'data' in the same node as sentiment_detector.py
- place the dataset inside
- from the node of sentiment_detector.py
- TYPE py sentiment_detector.py
- TYPE gather

# to run analysis
- TYPE py sentiment_detector.py
- TYPE run

# to do both together in one run
- TYPE py sentiment_detector.py
- TYPE full

