## Aspect Extraction using Topic Modelling on Amazon Phone Product Reviews

### Method
I used Latent Dirichlet Allocation (LDA), a probabilistic approach for topic modelling to do aspect extraction
on phone product reviews. To know about details of LDA,
please refer to [this page](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation).

The steps for aspect extraction are as follow:
1. Extract term frequency from given data
2. Construct the LDA model and train it.
3. Extract 5 topics from the dataset with 6 top words each.
4. Assign a topic for each review.
5. Evaluate the assignment by randomly sample 20 reviews from the dataset
    and manually annotate their topics. 
6. With the ground truth, calculate precision, recall and F1 score.
6. Find the average rating for each topic.

Source code in `lda-analysis.py`. I used sklearn's implementation for LDA.


### Result

Topics modelled using sklearn LDA -
```
Topic 0:
charger charge battery usb works cable
Topic 1:
case phone screen iphone great like
Topic 2:
headset ear sound bluetooth quality good
Topic 3:
just phone don like use product
Topic 4:
phone battery use phones device life
```

Topic 3 does not seem to be a promising topic. I removed this
topic, and for the reviews assigned to this topic I assigned a new
topic of the *second highest probability*.

With the final 4 topics, below shows the number of reviews and average ratings
on each phone aspect:

![](Number%20of%20Reviews%20&%20Average%20Ratings%20on%20Phone%20Aspects.png)

Overall, each of the aspect performed well. The ratings are all higher than 4.

The best performing aspect is battery and charging. The lowest is about the 
quality of the phone and its price. The most received reviews are about phone
cases and screen protectors.

### Evaluation

Confusion Matrix:

|     | Actual_0  | Actual_1 | Actual_2 | Actual_3 | Total |
|---  | :---------: | :---------:|:---------:| :---------: | :---------: |
| P_0 |3|0|1|0|4|
| P_1 |0|6|0|0|6|
| P_2 |0|3|2|1|6|
| P_4 |0|0|0|4|4|
|Total|3|9|3|5

Evaluation metrics:

| Topic | Precision  | Recall | F1 |
| :---------: | :---------: | :---------: | :---------:|
|0|0.75|1|0.857|
|1|1|0.66|0.8|
|2|0.33|0.66|0.44|
|4|1|0.8|0.88|

Obviously topic 2 yields a significantly lower F1 score. Reason is that topic
2 is itself somehow vague and any review could have contain the word "great",
"phone" and "price". So it is not a good topic to be used.

### References
1. https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
2. http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html