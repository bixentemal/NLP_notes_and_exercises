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

|D| : if documents as dimension
|V| : if words as dimension

By far the most common similarity metric is the cosine of the angle between the vectors.

The cosine is based on the dot product (also called the inner product)

$dot product(v,w) = v·w = \sum_{i=1}^{N}{v_{i}w_{j}}$

This raw dot product, however, has a problem as a similarity metric: it favors long vectors. 

The dot product is higher if a vector is longer, with higher values in each dimension. More frequent words have longer vectors, since they tend to co-occur with more words and have higher co-occurrence values with each of them. The raw dot product thus will be higher for frequent words. But this is a problem; we’d like a similarity metric that tells us how similar two words are regardless of their frequency.
The normalized dot product turns out to be the same as the cosine of the angle between the two vectors, following from the definition of the dot product between two vectors a and b:

a·b = |a||b|cosθ

$\frac{a·b}{|a||b|} = cosθ$

The cosine similarity metric between two vectors v and w thus can be computed as:

with vector len :

$ |v| = \sqrt{\sum_{i=1}^{N}{v_{i}^{2}}}  $

$ cosine(a, b) = \frac{a·b}{|a||b|}$

For some applications we pre-normalize each vector, by dividing it by its length, creating a <b>unit vector</b> of length 1. Thus we could compute a unit vector from a by dividing it by |a|. For unit vectors, the dot product is the same as the cosine.


## 6.5 TF-IDF: Weighing terms in the vector


The paradox lies in the fact that while words that occur frequently nearby (indicating contextual relevance, like "pie" near "cherry") are important, words that are too ubiquitous (like "the" or "good") lose their significance in providing meaningful context. So, there's a balance to be struck between frequency and specificity when determining the importance of words in analyzing text data.

=> usually used when the dimensions are documents

## 6.6 Pointwise Mutual Information (PMI)

An alternative weighting function to tf-idf, PPMI (positive pointwise mutual infor- mation), is used for term-term-matrices, when the vector dimensions correspond to words rather than documents. PPMI draws on the intuition that the best way to weigh the association between two words is to ask how much more the two words co-occur in our corpus than we would have a priori expected them to appear by chance.

## 6.7 Applications of the tf-idf or PPMI vector models

//

## 6.8 Word2vec

embeddings => short dense vectors
- embeddings are short, with number of dimensions d ranging from 50-1000
- the vectors are dense: instead of vector entries being sparse, mostly-zero counts or functions of counts, the values will be real-valued numbers that can be negative.

<b>skip-gram with negative sampling</b>, sometimes called <b>SGNS</b>. The skip-gram algorithm is one of two algorithms in a software package called <b>word2vec</b>.

Word2vec embeddings are static embeddings, meaning that the method learns <b>one fixed embedding for each word in the vocabulary</b> (unlike BERT which is dynamic contextual embeddings).

The intuition of word2vec is that instead of counting how often each word w occurs near, say, "apricot", we’ll instead train a classifier on a binary prediction task: “Is word w likely to show up near "apricot"?” We don’t actually care about this prediction task; instead we’ll take the learned classifier weights as the word embeddings.

word2vec = binary classifier

worf2vec = logistic regression classifier

assume we’re using a window of ±2 context words:

<pre>
[tablespoon of apricot jam,      a]
c1         c2    w     c3       c4
</pre>

Our goal is to train a classifier such that, given a tuple (w,c) of a target word w paired with a candidate context word c, it will return the probability that c is a real context word.

P(+|w, c)

ex
P(+|apricot, jam) > P(+|apricot, aardvark)

The probability that word c is not a real context word for w is just:

P(−|w, c) = 1 − P(+|w, c)

How does the classifier compute the probability P? The intuition of the skip- gram model is to base this probability on embedding similarity: a word is likely to occur near the target if its embedding vector is similar to the target embedding.

We use dot product to compute the similarity :

Similarity(w,c) ≈ c·w

and the sigoid function to convert the reslt to a probability

for a single word:

$P(+|w,c) = σ(c·w)= \frac{1}{1+exp(−c·w)}$

if there are many words, we assume all context words are independants by multiplying the probability of each.
for a context windows of L words :

$P(+|w,c_{1:L}) = \prod_{i=1}^{N}{σ(c·w_{i})}$

or

$log\ P(+|w,c_{1:L}) = \sum_{i=1}^{N}{log\ σ(c·w_{i})}$

## 6.8.2 Learning skip-gram embeddings

Strategy for L=+-2 and k=2 

for each w, [$c_{1}$, $c_{2}$, $c_{3}$, $c_{4}$] :

1. create 4 positive examples +, and 4xk=4x2=8 negativ examples (ajusted by weighted unigram, parameter $\alpha$=0.75)
2. We need to maximize the similarity between c and positive examples, and minimise the similarity with negative examples. We thus create a loss function

$L_{CE} = -log[P(+|w, c_{pos})\prod_{i=1}^kP(-|w,c_{neg_{i}})] $

$L_{CE} = -[log\ σ(w · c_{pos})\sum_{i=1}^klog\ σ(-c_{neg_{i}} · w)] $

3. We minimize this loss function using stochastic gradient descent with randomly ini- tialized W and C matrices

4. he skip-gram model learns two separate embeddings for each word i: the target embedding w and the context embedding c , stored in two matrices, the target matrix W and the context matrix C. It’s common to just add them together, representing word i with the vector wi +ci.

## 6.8.3 Other kinds of static embeddings

fasttext addresses a problem with word2vec as we have presented it so far: it has no good way to deal with unknown words

Fasttext deals with these problems by using subword models, representing each word as itself plus a bag of constituent n-grams

https://fasttext.cc.

Another very widely used static embedding model is GloVe. GloVe is based on ratios of probabilities from the word-word cooccurrence matrix, combining the intuitions of count-based models like PPMI while also capturing the linear structures used by methods like word2vec.

# 6.9 Visualizing Embeddings

Method1 : The simplest way to visualize the meaning of a word w embedded in a space is to list the most similar words to w by sorting the vectors for all words in the vocabulary by their cosine with the vector for w. For example the 7 closest words to frog using a particular embeddings computed with the GloVe algorithm are: frogs, toad, litoria, leptodactyli- dae, rana, lizard, and eleutherodactylus

Method2 : t-SNE

# 6.10 Semantic properties of embeddings

first-order co-occurrence (sometimes called syntagmatic association) if they are typically nearby each other.

second-order co-occurrence (sometimes called paradigmatic association) if they have similar neighbors

Another semantic property of embeddings is their ability to capture relational meaning: parallelogram model

Paris − France + Italy results #»
in a vector that is close to Rome.

## 6.10.1 Embeddings and Historical Semantics

Embeddings can also be a useful tool for studying how meaning changes over time,
by computing multiple embedding spaces, each from texts written in a particular time period. 

# 6.11 Bias and Embeddings

embeddings can roughly model relational similar- ity: ‘queen’ as the closest word to ‘king’ - ‘man’ + ‘woman’ implies the analogy
There are also a number of recent works analyzing semantic change using computational
man:woman::king:queen. But these same embedding analogies also exhibit gender methods. 

allocational harm, when a system allo- information-based embeddings and found that semantic changes uncovered by their
cates resources (jobs or credit) unfairly to different groups. 

//

# 6.12 Evaluating Vector Models

The most important evaluation metric for vector models is extrinsic evaluation on tasks, i.e., using vectors in an NLP task and seeing whether this improves perfor- mance over some other model.

Nonetheless it is useful to have intrinsic evaluations. The most common metric is to test their performance on similarity, computing the correlation between an algorithm’s word similarity scores and word similarity ratings assigned by humans.

//

# 6.13 Summary

• In vector semantics, a word is modeled as a vector — a point in high-dimensional space, also called an embedding. In this chapter we focus on static embeddings, where each word is mapped to a fixed embedding.

• Vector semantic models fall into two classes: sparse and dense. In sparse models each dimension corresponds to a word in the vocabulary V and cells are functions of co-occurrence counts. The term-document matrix has a row for each word (term) in the vocabulary and a column for each document. The word-context or term-term matrix has a row for each (target) word in the vocabulary and a column for each context term in the vocabulary. Two sparse weightings are common: the tf-idf weighting which weights each cell by its term frequency and inverse document frequency, and PPMI (point- wise positive mutual information), which is most common for word-context matrices.

• Dense vector models have dimensionality 50–1000. Word2vec algorithms like skip-gram are a popular way to compute dense embeddings. Skip-gram trains a logistic regression classifier to compute the probability that two words are ‘likely to occur nearby in text’. This probability is computed from the dot product between the embeddings for the two words

• Skip-gram uses stochastic gradient descent to train the classifier, by learning embeddings that have a high dot product with embeddings of words that occur nearby and a low dot product with noise words.


• Other important embedding algorithms include GloVe, a method based on ratios of word co-occurrence probabilities.

• Whether using sparse or dense vectors, word and document similarities are computed by some function of the dot product between vectors. The cosine of two vectors—a normalized dot product—is the most popular such metric.

