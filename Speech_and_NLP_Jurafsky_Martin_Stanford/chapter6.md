---
title: 6. Vector Semantics and Embeddings
subtitle: >+
  Speech and Language Processing. Daniel Jurafsky & James H. Martin. Copyright ©
  2023. All rights reserved. Draft of February 3, 2024.


  Source : https://web.stanford.edu/~jurafsky/slp3/

---

distributional hypothesis = "words which are synonyms (like oculist and eye-doctor) tended to occur in the same environment".

These word representations are also the first example in this book of repre- sentation learning,

Semantic fields, topic models, like Latent Dirichlet Allocation, LDA, which apply unsupervised learning on large sets of texts to induce sets of associated words from text.

A semantic frame is a set of words that denote perspectives or participants in a particular type of event.

Connotation : aspects of a word’s meaning that are related to a writer or reader’s emotions.

Positive or negative evaluation language is called sentiment. Early work on affective meaning (Osgood et al., 1957) found that words varied along three important dimensions of affective meaning:


- valence: the pleasantness of the stimulus

- arousal: the intensity of emotion provoked by the stimulus dominance: the degree of 

- control exerted by the stimulus

Thus words like happy or satisfied are high on valence, while unhappy or an- noyed are low on valence. Excited is high on arousal, while calm is low on arousal. Controlling is high on dominance, while awed or influenced are low on dominance.


In this chapter we’ll introduce the two most commonly used models :

- <b>tf-idf model</b>, an important baseline, the meaning of a word is defined by a simple function of the counts of nearby words.

- <b>word2vec</b> model family for construct- ing short, dense vectors that have useful semantic properties

# 6.3 Words and Vectors

## 6.3.1 Words and Vectors

term-document matrix

        As You Like It  Twelfth Night   Julius  Caesar Henry V
battle          1               0           7           13
good            114             80          62          89
fool            36              58          1           4
wit             20              15          2           3

each word is represented by a row vector :
ex : battle is [1, 0, 7, 13]

=> most of these numbers are zero these are sparse vector representations

Information retrieval (IR) is the task of finding the document d from the D documents in some collection that best matches a query q.

## 6.3.2 Words as vectors: document dimensions

row vector

## 6.3.3 Words as vectors: word dimensions

word-word matrix

## 6.4 Cosine for measuring similarity

To measure similarity between two target words v and w

|D| : if documnts as dimension
|V| : if words as dimension

By far the most common similarity metric is the cosine of the angle between the vectors.

The cosine is based on the dot product (also called the inner product)

$dot product(v,w) = v·w = \sum_{i=1}^{N}{v_{i}w_{j}}$

This raw dot product, however, has a problem as a similarity metric: it favors long vectors. 

The dot product is higher if a vector is longer, with higher values in each dimension. More frequent words have longer vectors, since they tend to co-occur with more words and have higher co-occurrence values with each of them. The raw dot product thus will be higher for frequent words. But this is a problem; we’d like a similarity metric that tells us how similar two words are regardless of their frequency.
The normalized dot product turns out to be the same as the cosine of the angle between the two vectors, following from the definition of the dot product between two vectors a and b:

a·b = |a||b|cosθ

The cosine similarity metric between two vectors v and w thus can be computed as:

with vector len :

$ |v| = \sqrt{\sum_{i=1}^{N}{v_{i}^{2}}}  $

$ cosine(a, b) = \frac{a·b}{|a||b|}$

For some applications we pre-normalize each vector, by dividing it by its length, creating a <b>unit vector</b> of length 1. Thus we could compute a unit vector from a by dividing it by |a|. For unit vectors, the dot product is the same as the cosine.


## 6.5 TF-IDF: Weighing terms in the vector

