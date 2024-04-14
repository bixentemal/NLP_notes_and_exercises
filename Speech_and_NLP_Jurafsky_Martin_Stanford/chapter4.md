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

- all words are equivalents (naive Bayes assumption)
- word position doesn't matter (bag of words)

= probabilistic classifier.

for a document d, out of all classes c ∈ C the classifier returns the class cˆ which has the maximum posterior ˆ probability given the document.
 
cˆ = argmaxP(c|d)

The intuition of Bayesian classification is to use Bayes’ rule to transform the equation above into other probabilities that have some useful properties.

$P(y|c)=\frac{P(y|x)P(x)}{P(y)}$

$cˆ = argmax P(c|d) = argmax \frac{P(d|c)P(c)}{P(d)}$

We cn drop the denominator since P(d) doesn't change for each class (always the same docment we try to find the class)

$cˆ = argmax P(d|c)P(c)$

To return to classification: we compute the most probable class cˆ given some document d by choosing the class which has the highest product of two probabilities: the prior probability of the class P(c) and the likelihood of the document P(d|c).

$cˆ=argmaxP(f1,f2,....,fn|c) P(c)$

Given the <b>naive Bayes assumption</b> we can simplify to 

$P(f1, f2,...., fn|c) = P(f1|c)·P(f2|c)·...·P(fn|c)$

The final equation for the class chosen by a naive Bayes classifier is thus:

$C_{NB} = argmax P(c) \prod_{i = 1}^{position} P(w_i|c)$

re done in log space, to avoid underflow and increase speed

$C_{NB} = argmax\ log P(c) \sum_{i = 1}^{position} log\ P(w_i|c)$

By considering features in log space -> linear classifiers.

# 4.2 Training the Naive Bayes Classifier

How can we learn the probabilities $P(c)$ and $P(fi|c)$ :

- for the maximum likelihood estimate, we’ll simply use the frequencies in the data. 

feature = existence of a word in the document’s bag of words.

We first concatenate all documents with category c into one big “category c” text.

$P(W_i|c) = \frac{count(w_i,c)}{\sum_{w \in V}count(w,c)}$

Since one of the probability can be 0 in the training set, we need to apply smoothing.

$P(W_i|c) = \frac{count(w_i,c)+1}{(\sum_{w \in V}count(w,c))+V}$

V consists of the union of all the word types in all classes, not just the words in one class c.

Finally, some systems choose to completely ignore another class of words: stop words, very frequent words like the and a. This can be done by sorting the vocabulary by frequency in the training set, and defining the top 10–100 vocabulary entries as stop words, or alternatively by using one of the many predefined stop word lists available online. Then each instance of these stop words is simply removed from both training and test documents as if it had never occurred. In most text classifica- tion applications, however, using a stop word list doesn’t improve performance, and so it is more common to make use of the entire vocabulary and not use a stop word list.

- for the class prior P(c) we ask what percentage of the documents in our training set are in each class c. Let Nc be the number of documents in our training data with class c and Ndoc be the total number of documents. Then: 

$P^ \ (c) = \frac{N_c}{N_{doc}}$


