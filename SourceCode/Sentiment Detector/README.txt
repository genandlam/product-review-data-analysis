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

========================================================================================
OUTPUT
========================================================================================

To keep the output simple, the tokens which make up the stem is written to a separate 
text file, in results.txt

Below is an example of the tokens which make up the stem 'great' from results.txt
The semicolons are not part of the token!
========================================================================================
#1 : great    value = 364040
    great;
    great.;
    ..great;
    greatly.;
    great..;
    *great*;
    greatness.;
    great-;
    -great;
    great*;
    greatful;
    .great;
    greats;
    greate;
    greatness;
    great/;
    ***great***;
    greats.;
    greatly;

========================================================================================
CONSOLE OUTPUT
========================================================================================

goal mean: the intended normalized mean of the score
document_norm_mean: the actual normalized mean 
value: the derived score for the particular stem

========================================================================================
goal_norm_mean: 0.040000000000000036 , document_norm_mean: 0.6
{5: 1.1, 4: 0.6600000000000001, 3: 0.22000000000000003, 2: -2.34, 1: -3.9000000000000004}

Top 20 Positive Sentiment Words
#1 : great    value = 364040
#2 : good    value = 171847
#3 : love    value = 171285
#4 : use    value = 164739
#5 : charg    value = 123026
#6 : nice    value = 118705
#7 : easi    value = 84906.7
#8 : best    value = 82853.4
#9 : need    value = 74174.1
#10 : perfect    value = 73076
#11 : littl    value = 68379
#12 : look    value = 65356.3
#13 : protect    value = 63209.6
#14 : usb    value = 60949.5
#15 : recommend    value = 55427.3
#16 : excel    value = 54394.5
#17 : work    value = 53387.5
#18 : awesom    value = 41684.5
#19 : keep    value = 39884.9
#20 : want    value = 39181.3

Top 20 Negative Sentiment Words
#1 : poor    value = -31183.3
#2 : return    value = -29984.2
#3 : disappoint    value = -22437.1
#4 : broke    value = -20482.7
#5 : cheap    value = -19067.4
#6 : horribl    value = -18537
#7 : terribl    value = -18366.8
#8 : bad    value = -18289.8
#9 : wast    value = -15643.3
#10 : stop    value = -13372.2
#11 : pay    value = -10794
#12 : worst    value = -10390.2
#13 : send    value = -8922.71
#14 : defect    value = -8774.63
#15 : useless    value = -7904.36
#16 : fell    value = -6386.65
#17 : sent    value = -6064.02
#18 : broken    value = -5970.47
#19 : fail    value = -5752.77
#20 : wors    value = -4903.03