# Aspect Extraction on Amazon Phone Product Reviews

### Method
I used a Latent Dirichlet Allocation (LDA), a state-of-the-art 
probabilistic approach for topic modelling to do aspect extraction
on phone product reviews.

File `nmf-test.py` uses sklearn's implementation for both.
I see that the runtime is faster, and NMF gives more pleasing topic models.
But there is no way for visualization.

On the other hand `topic_analysis.py` runs LDA using gensim.
It's much more slower and the topics don't look good.
The only good thing is it shows a great visualization.

Results: (ran on first 20k examples)

Topics modelled using sklearn NMF -
```
Topic 0:
charger works great car usb cable -> charger
Topic 1:
case iphone phone cases protection fit -> phone cases
Topic 2:
headset ear sound bluetooth quality good -> headphones
Topic 3:
screen protector protectors bubbles product iphone -> screen protectors
Topic 4:
phone battery charge use cell new -> battery charge
```

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

Topics modelled using gensim LDA:   
```
(0, '0.026*"headset" + 0.019*"sound" + 0.017*"bluetooth" + 0.015*"phone" + 0.013*"quality" + 0.009*"button"')
(1, '0.031*"phone" + 0.021*"screen" + 0.016*"great" + 0.015*"iphone" + 0.014*"product" + 0.013*"would"')
(2, '0.043*"phone" + 0.033*"charge" + 0.024*"battery" + 0.019*"charger" + 0.012*"cable" + 0.009*"iphone"')
```

Visualization:
![](t1.PNG)

Planning for next steps:
- [ ] Run on the full dataset
- [ ] Choose the best topic model (currently is NMF, the topics are distinctive enough).
- [ ] Run topic modelling on each document and do F1 scoring.
- [ ] Run polarity check on each review based on its topic, and do an overall analysis.
      By this we can inspect how well Amazon's cell phone products do in each topic aspect.