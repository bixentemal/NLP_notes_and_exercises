---
title: 3.N-gram Language Models
subtitle: Speech and Language Processing. Daniel Jurafsky & James H. Martin.
  Copyright © 2023. All rights reserved. Draft of February 3, 2024.


  Source : https://web.stanford.edu/~jurafsky/slp3/
---

n-gram :
- is the simplest kind of language model
- can mean a probabilistic model that can estimate the probability of a word given the n-1 previous words.
- Can also be used to to assign probabilities to entire sequences.
- != neural large language models (i.e transformers), but usefull to introduce concepts (like training and test sets, perplexity, sampling, and interpolation).

an n-gram is a sequence of n words: a 2-gram (which we’ll call bigram) is a two-word sequence of words like “please turn”

# 3.1 N-Grams

P(w|h) = the probability of a word w given some history h.

can be stated as : "Out of the times we saw the history h, how many times was it followed by the word w”, as follows:

$\
P(the|its water is so transparent that) = 
$
$\frac{C(its water is so transparent that the)}{C(its water is so transparent that)}
$

Word count is not easy to manage (especially in a very large corpus like the web).

### Formalization :

For the joint probability of each word in a se- quence having a particular value P(X1 = w1,X2 = w2,X3 = w3,...,Xn = wn) we’ll use P(w1,w2,...,wn).

chain rule of probability:

P(X1...Xn) = P(X1) P(X2|X1) P(X3|X1:2) ... P(Xn|X1:n−1) 

= $\prod_{k = 1}^{n} P(Xk|X1:k−1)$


 we could estimate the joint probability of an entire sequence of words by multiplying together a number of conditional probabilities.

 The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words.

The assumption that the probability of a word depends only on the previous word is called a <b>Markov</b> assumption.

Markov models are the class of probabilistic models that assume we can predict the probability of some future unit without looking too far into the past.

General approximation :

We’ll use N here to mean the n-gram size, so N = 2 means bigrams and N = 3 means trigrams.

$P(w_{n} \mid w_{1:n-1}) \sim P(w_{n} \mid w_{n-N+1:n-1}) $

### maximum likelihood estimation or MLE

// TO BE CONTINUED