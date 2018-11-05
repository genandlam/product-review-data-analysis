## Product Review Data Analysis

An NLP project for CZ4045 - Natural Language Processing by Group 18.

The 4 parts for this project include:
1. Dataset analysis
2. Building a noun phrase detector
3. Sentiment analysis on reviews
4. Application - Aspect extraction using topic modelling on reviews

 ==================
| Dataset Analysis |
 ==================
Libraries and installation:
```
cd SourceCode/Dataset Analysis/
pip install nltk
pip install pandas
pip install matplotlib
```

Run `python DatasetAnalysis.py` for results.
Please refer to SouceCode/Dataset Analysis/README.md for sample output.

 ======================
| Noun Phrase Detector |
 ======================
Libraries and installation:
```
cd SourceCode/Noun Phrase Detector
pip install nltk
pip install textblob
```
Run `python top_20_noun_phrases.py` for results.
Please refer to SouceCode/Noun Phrase Detector/README.md for sample output.

 ====================
| Sentiment Detector |
 ====================

I. INSTALLING LIBRARIES
Create a virtual environment.
- on cmd, navigate to your desired directory
- TYPE py -m pip install virtualenv  #'py' calls python executable, may be 'python' depending on installation
- TYPE py -m virtualenv NLP          # NLP can be named arbitrarily
- TYPE NLP\Scripts\Activate
- TYPE pip install nltk
- TYPE pip install numpy
- TYPE pip install tqdm

II. RUNNING PROGRAM
-on cmd, activate virtual environment created prior
-navigate to the folder with sentiment_detector.py

# to collect tokens and data, 
- create a folder named 'data' in the same node as sentiment_detector.py
- place the dataset inside the 'data' folder
- from the node of sentiment_detector.py
- TYPE py sentiment_detector.py
- TYPE gather

# to run analysis
- TYPE py sentiment_detector.py
- TYPE run

# to do both together in one run
- TYPE py sentiment_detector.py
- TYPE full

Please refer to SourceCode/Sentiment Detector/README.txt for sample output.

 =================
| Topic Modelling |
 =================
Libraries and installation:
```
cd SourceCode/Topic Modelling/
pip install sklearn
pip install tqdm
pip install numpy
```
Run `python lda-analysis.py` for results.
Please refer to SourceCode/Topic Modelling/README.md for sample output.

