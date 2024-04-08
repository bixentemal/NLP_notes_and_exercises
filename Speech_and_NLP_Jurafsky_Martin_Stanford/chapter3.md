---
title: 3.N-gram Language Models
subtitle: Speech and Language Processing. Daniel Jurafsky & James H. Martin.
  Copyright © 2023. All rights reserved. Draft of February 3, 2024.


  Source : https://web.stanford.edu/~jurafsky/slp3/
---

n-gram :
- is the simplest kind of language model
- probabilistic model that can estimate the probability of a word given the n-1 previous words.
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

The bigram model <b>approximates the probability of a word given all the previous words</b> P(wn|w1:n−1) by using only the conditional probability of the preceding word P(wn|wn−1). 

 The intuition of the n-gram model is that instead of computing the probability of a word given its entire history, we can approximate the history by just the last few words.

The assumption that the probability of a word depends only on the previous word is called a <b>Markov</b> assumption.

Markov models are the class of probabilistic models that assume we can predict the probability of some future unit without looking too far into the past.

General approximation :

We’ll use N here to mean the n-gram size, so N = 2 means bigrams and N = 3 means trigrams.

$P(w_{n} \mid w_{1:n-1}) \sim P(w_{n} \mid w_{n-N+1:n-1}) $

### maximum likelihood estimation or MLE

MLE estimate for the parameters of an n-gram model by getting counts from a corpus, and normalizing the counts so that they lie between 0 and 1

For bigram model :

$P(w_{n} \mid w_{1:n-1}) =
$
$\frac{C( w_{n-1}w_{n})}{\sum_{w} C( w_{n-1}w_)}
$

We can generalize for N-gram model :

$P(w_{n} \mid w_{1-N+1:n-1}) =
$
$\frac{C( w_{n-N+1}w_{n})}{\sum_{w} C( w_{n-N+1}w_)}
$

This ratio is called a <b>relative frequency</b>. 

<pre>
                                   ┌────────────────────┐                                       
                                   │   Language model   │                                       
                                   └────────────────────┘                                       
                                              △                                                 
                                              │                                                 
                                   ┌────────────────────┐                                       
                                   │       N-GRAM       │                                       
                                   └───┬────────────────┘                                       
               Markov assumption       │                                                        
                                  ─────┘                                                        
                                 △                                                              
                                 │                                                              
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐      ┌────────────────────┐
│       1-GRAM       │ │       2-GRAM       │ │       3-GRAM       │      │       N-GRAM       │
│     (unigram)      │ │                    │ │                    │ ...  │                    │
└────────────────────┘ └────────────────────┘ └────────────────────┘      └────────────────────┘
                                              │                                                 
                                              │                                                 
                                              │ to estimate probabilities of a n-gram           
                                              │              we use MLE                         
                                              │                                                 
                                              ▼                                                 
                                                                                                
                                    Estimated using MLE                                         
                                 (counts from a corpus, and                                     
                               normalizing the counts so that                                   
                                 they lie between 0 and 1)                                      
                                                                                                
                                                                                                

</pre>

Practical example for bigram :

```
P(i|<s>) = 0.25 
P(english|want) = 0.0011
P(food|english) = 0.5 
P(</s>|food) = 0.68
```
probability of the complete sentence <pre> i want english food </pre>
can be calculated this way
```
P(<s> i want english food </s>)
= P(i|<s>) x P(want|i) x P(english|want) x P(food|english)x P(</s>|food) 
= .25×.33×.0011×0.5×0.68
= .000031
```

We always represent and compute language model probabilities in log format as <b>log probabilities</b> since the more probabilities we multiply together, the smaller the product becomes.

p1×p2×p3×p4 =exp(logp1+logp2+logp3+logp4)

# 3.2 Training and test set

<pre>
                                                                                                  
                                              Evaluated once at the very end                      
                                                                                                  
                                   ┌───────────────────────────────────────────────────┐          
                                   │                                                   │          
                                   │                                                   ▼          
┌────────────────────┐      ┌────────────┐        ┌────────────────────┐    ┌────────────────────┐
│    TRAINING SET    │      │   Model    │        │  DEVELOPMENT SET   │    │      TEST SET      │
│                    ◀──────▶            │        │                    │    │                    │
└────────────────────┘      └────────────┘        └────────────────────┘    └────────────────────┘
                                   │                         ▲                                    
                                   │                         │                                    
                                   └─────────────────────────┘                                    
                                                                                                  
                                 Multiple testing                                    
                                                                                                  
</pre>

Due to multiple testing on Dev set, we create an adherence, this introduces a bias that makes the probabilities all look too high, and causes huge inaccuracies in <<b>perplexity </b>


# 3.3 Evaluating Language Models: Perplexity

The perplexity (sometimes abbreviated as PP or PPL) of a language model on a test set is the inverse probability of the test set (one over the probability of the test set), normalized by the number of words N(i.e per word perplexity)

For a test set W = w1 w2 . . . wN ,:

perplexity(W) = $\sqrt[N]{\prod_{i = 1}^{N}\frac{1}{P(w_n|w_1... w_i-1)}}$

<b>Perplexity has an inverse relationship with probability => the lower the perplexity, the better the model</b>. 

So a lower perplexity can tell us that a language model is a better predictor of the words in the test set.

since this sequence will cross many sentence boundaries, if our vocabulary includes a between-sentence token \<EOS\> or separate begin- and end-sentence markers \<s\> and \</s\> then we can include them in the probability computation. If we do, then we also include one token per sentence in the total count of word tokens N.

# 3.4 SAMPLING SENTENCES FROM A LANGUAGE MODEL 11

Sampling from a distribution means to choose random points according to their likelihood

= technique of visualizing

Example for unigram :
Imagine all the words of the English language covering the probability space between 0 and 1, each word covering an interval proportional to its frequency.
We choose a random value between 0 and 1, find that point on the probability line, and print the word whose interval includes this chosen value. We continue choosing random numbers and generating words until we randomly generate the sentence-final token \</s\>.


<pre>
                                                                                    
         the               of          a       to      in                           
┌────────────────────┬─────────────┬────────┬───────┬───────┐                       
│        0.06        │    0.03     │  0.02  │ 0.02  │ 0.02  │                       
└────────────────────┴─────────────┴────────┴───────┴───────┘                       
                                                                                    
│                    │             │        │       │           ...                 
└────────────────────┴─────────────┴────────┴───────┴─────────                      
                    .06           .09      .11     .13      cumulative probability  
</pre>

# 3.5 Generalization and Zeros

The n-gram model depends on the training corpus.

Another implication is that n-grams do a better and better job of modeling the training corpus as we increase the value of N.

Exemple of sampling on a model generated from the shakespare corpus for unigram, 2-gram, 3-gram, 4-gram respectivelly :

<pre>
1-gram
–To him swallowed confess hear both. Which. Of save on trail for are ay device and
rote life have
–Hill he late speaks; or! a more to leg less first you enter

2-gram
–Why dost stand forth thy canopy, forsooth; he is this palpable hit the King Henry. Live king. Follow.
gram –What means, sir. I confess she? then all sorts, he is trim, captain.

3-gram
–Fly, and will rid me these news of price. Therefore the sadness of parting, as they say,’tis done.
gram –This shall forbid it should be branded, if renown made it empty.

4-gram
–King Henry. What! I will go seek the traitor Gloucester. Exeunt some of the watch. A
great banquet serv’d in; gram –It cannot be but so.

</pre>

The more the n (in n-gram), the more coherent the sampling result,(and the more sparse the matrix)

another similar samplong example with the  Wall Street Journal

<pre>
1 gram
Months the my and issue of year foreign new exchange’s september
were recession exchange new endorsed a acquire to six executives

2
Last December through the way to preserve the Hudson corporation N.
B. E. C. Taylor would seem to complete the major central planners one gram point five percent of U. S. E. has already old M. X. corporation of living
on information such as more frequently fishing to keep her

3
They also point to ninety nine point six billion dollars from two hundred four oh six three percent of the rates of interest stores as Mexico and
gram Brazil on market conditions
</pre>

=> comparing both, we can see the adherence to the training corpus

conclusion :

To build a language model for translating legal documents, we need a training corpus of legal documents. 

To build a language model for a question-answering system, we need a training corpus of questions.

etc...

It is equally important to get training data in the appropriate dialect or variety, especially when processing social media posts or spoken transcripts.

<b>sparsity</b> isa problem : if a n-gram is not present in the training set but present on the test set, some evaluation which are syntaxically correct will be returned with a P(x|y) = 0 on the test set.

These <b>zeros</b>—things that don’t ever occur in the training set but do occur in the test set—are a problem for two reasons.

- First, their presence means we are underestimating the probability of all sorts of words that might occur, which will hurt the performance of any application we want to run on this data.

- Second, if the probability of any word in the test set is 0, the entire probability of the test set is 0. By definition, perplexity is based on the inverse probability of the
closed vocabulary
test set. Thus if some words have zero probability, we can’t compute perplexity at all, since we can’t divide by 0!

2 kinds of zeros :
- n-gram probability is zero because they occur in a novel test set context -> <b>smoothing</b> algorithm
- unknown words

### Unknown Words

2 cases
- <b>closed vocabulary</b> : no unknown words (since the space of possible words are known in advance)
- <b>open vocabulary</b> : in this case we can have <b>out of vocabulary (OOV) words</b>

The percentage of OOV words that appear in the test set is called the <b>OOV rate</b>.

One way to close and open vocabulary system is to :
1. Choose a vocabulary (word list) that is fixed in advance.
2. Convert in the training set any word that is not in this set (any OOV word) to the unknown word token <UNK> in a text normalization step.
3. Estimate the probabilities for <UNK>from its counts just like any other regular word in the training set.

# 3.6 Smoothing

smoothing, also called discounting

## 3.6.1 Laplace Smoothing

The simplest way to do smoothing is to add one to all the n-gram counts, before we normalize them into probabilities. All the counts that used to be zero will now have a count of 1, the counts of 1 will be 2, and so on. This algorithm is called Laplace smoothing.

Here is an <b>unsmoothed</b> maximum likelihood estimate of the unigram probability of the word wi is its count ci normalized by the total number of word tokens N :

$P({w}_i) = \frac{{c}_i}{N}$

Now applying Laplace smoothing (also called add-one smoothing)

${P}_{Laplace}({w}_i) = \frac{{c}_i+1}{N+V}$

Adjusted count c∗ is easier to compare directly with the MLE counts and can be turned into a probability like an MLE count by normalizing by N

${c}_i^* = ({c}_i+1)\frac{N}{N+V}$

From this we can create a relative discount dc, the ratio of the discounted counts to the original counts:

${d}_c=\frac{{c}^*}{c}$

## 3.6.2 Add-k smoothing

laplace smoothing = Add-k smoothing, with k=1

${P}^*_{Add-k}({w}_i) = \frac{C({w}_{n-1}{w}_{n})+k}{C({w}_{n-1})+kV}$

## 3.6.3 Backoff and Interpolation



# 3.7 Huge Language Models and Stupid Backoff