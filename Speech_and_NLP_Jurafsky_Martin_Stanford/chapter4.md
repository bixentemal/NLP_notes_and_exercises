---
title: 4.Naive Bayes, Text Classification, and Sentiment
subtitle: Speech and Language Processing. Daniel Jurafsky & James H. Martin.
  Copyright © 2023. All rights reserved. Draft of February 3, 2024.


  Source : https://web.stanford.edu/~jurafsky/slp3/
---

Classification :

- text categorization

- sentiment analysis

- Spam detection

- language detection

- authorship attribution

The most common way of doing text classification in language processing is instead via <b>supervised machine learning</b>

=> input   : x + a set of classes Y = {y1,y2,...,yM}

<= output  : predicted class y ∈ Y

A <b>probabilistic classifier</b> additionally will tell us the probability of the observation being in the class.

2 ways of doing classification :

- <b>Generative classifiers</b> like <b>naive Bayes</b> build a model of how a class could generate some input data. Given an observation, they return the class most likely to have generated the observation. 

- Discriminative classifiers like <b>logistic regression</b> instead learn what features from the input are most useful to discriminate between the different possible classes.

# 4.1 Naive Bayes Classifiers

multinomial naive Bayes classifier. Makes some naive assumptions :

- all words are equivalents (bag of words)
- word position doesn't matter

= probabilistic classifier.

for a document d, out of all classes c ∈ C the classifier returns the class cˆ which has the maximum posterior ˆ probability given the document.
 
cˆ = argmaxP(c|d)

The intuition of Bayesian classification is to use Bayes’ rule to transform the equation above into other probabilities that have some useful properties.

$P(y|c)=\frac{P(y|x)P(x)}{P(y)}$

$cˆ = argmax P(c|d) = argmax \frac{P(d|c)P(c)}{P(d)}$

We cn drop the denominator since P(d) doesn't change for each class (always the same docment we try to find the class)

$cˆ = argmax P(d|c)P(c)$
