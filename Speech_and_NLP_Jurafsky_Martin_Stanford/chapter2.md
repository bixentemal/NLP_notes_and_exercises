---
title: 2. Regular Expressions, Text Normalization, Edit Distance
subtitle: >+
  Speech and Language Processing. Daniel Jurafsky & James H. Martin. Copyright ©
  2023. All rights reserved. Draft of February 3, 2024.


  Source : https://web.stanford.edu/~jurafsky/slp3/

---

## Definitions

Text normalization : set of tasks targetting makeing text more "standard"

In text normalization, we have :

Tokenization : words are not token (ex : New York) and tokens are not words (ex : :-) )

lemmatization : consists of determining if two words have the same root (ex : sang, sing, song)

stemming :  simpler version of lemmization (strip suffixes from the end of the word)

sentence segmentation : for instance using , or .

edit distance : metric which mesures of how similar are two strings (= number of edits deletions or substitutions)

## 2.1 Regexps

// pass

## 2.2 Words

Corpus : collection of text
Examples : 
  - Brown corpus : million word collections of samples (Brown uiversity 1063/64
  
Punctuations like . or , allows to find words boundaries.

The Switchboard corpus of American English telephone conversations between strangers was collected in the early 1990s; it contains 2430 conversations averaging 6 minutes each, totaling 240 hours of speech and about 3 million words

utterance : is the spoken correlate of a sentence:

example : 
I do uh main- mainly business data processing

2 kinds of disfluencies here : fragments (like main), and filers like uh.

word types : number of (distincts) words in a corpus

word instances : number of "running" words

example : 

They picnicked by the pool, then lay back on the grass and looked at the stars.

-> 6 instances and 14 types

Example of corpus

<pre>
┌─────────────────────────────────────────────────────────────┐
│     Corpus              Instances = N       Types = |V|     │
├─────────────────────────────────────────────────────────────┤
│     Shakespeare         884thousand         31thousand      │
├─────────────────────────────────────────────────────────────┤
│     Browncorpus             1million        38thousand      │
├─────────────────────────────────────────────────────────────┤
│                             ...                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
</pre>

Herdan’s Law (Herdan, 1960) or Heaps’ Law :

|V| = kN^β

0 < β < 1
The value of β depends on the corpus size and the genre

the vocabulary size for a text goes up significantly faster than the square root of its length in words.

wordforms and lemma

cat and cats ae two wordforms whereas the lemma for both is cat

## 2.3 Corpora

 a specific dialect of a specific language, at a specific time, in a specific place, for a specific function.

 African American English (AAE)
 Mainstream American English (MAE)

 The datasheet of a particular corpus :

- Motivation: Why was the corpus collected, by whom, and who funded it?
Situation: When and in what situation was the text written/spoken? For example, was there a task? Was the language originally spoken conversation, edited text, social media communication, monologue vs. dialogue?
- Language variety: What language (including dialect/region) was the corpus in?
datasheet
- Speaker demographics: What was, e.g., the age or gender of the text’s authors?
- Collection process: How big is the data? If it is a subsample how was it sampled? Was the data collected with consent? How was the data pre-processed, and what metadata is available?
Annotation process: What are the annotations, what are the demographics of the annotators, how were they trained, how was the data annotated?
- Distribution: Are there copyright or other intellectual property restrictions?

## 2.4 Simple Unix Tools for Word Tokenization

tr -sc 'A-Za-z' '\n' < sh.txt | tr A-Z a-z | sort | uniq -c | sort -n -r
The results show that the most frequent words in Shakespeare, as in any other corpus, are the short function words like articles, pronouns, prepositions:
<pre>
27378 the
26084 and
22538 i
19771 to
17481 of
14725 a
13826 you
12489 my
11318 that
11112 in
...
</pre>

## 2.5 Word Tokenization

### Top-down (rule-based) tokenization : we define a standard and implement rules to implement that kind of tokenization.

Penn Treebank to- kenization standard, used for the parsed corpora (treebanks) released by the Lin- guistic Data Consortium (LDC).

based on regular expressions

Example : 
<pre>
>>> text = 'That U.S.A. poster-print costs $12.40...'
>>> pattern = r'''(?x)
...     (?:[A-Z]\.)+
...   | \w+(?:-\w+)*
...   | \$?\d+(?:\.\d+)?%?  # currency, percentages, e.g. $12.40, 82%
...   | \.\.\.            # ellipsis
... | [][.,;"'?():_`-] # these are separate tokens; includes ], [ ... '''
>>> nltk.regexp_tokenize(text, pattern)
['That', 'U.S.A.', 'poster-print', 'costs', '$12.40', '...']
</pre>

### Bottom-up tokenization :  we use simple statistics of letter sequences to break up words into subword tokens.

Ex : Byte-Pair Encoding, unigram language modeling , SentencePiece

Necessity to cut in subwords :

Most tokenization schemes have two parts: a token learner, and a token seg- menter.

<pre>
function BYTE-PAIR ENCODING(strings C, number of merges k) returns vocab V
V←all unique characters in C           # initial set of tokens is characters 

for i=1 to k do                        #merge tokens k times
tL, tR ←Most frequent pair of adjacent tokens in C
tNEW ←tL + tR     # make new token by concatenating V←V + tNEW # update the vocabulary
Replace each occurrence of tL, tR in C with tNEW # and update the corpus
return V
</pre>

## 2.6 Word Normalization, Lemmatization and Stemming

Word normalization is the task of putting words/tokens in a standard format (ex : case folding)

Lemmatization : we want two morphologically different forms of a word to behave similarly. Exampel in Polish : Warsaw / Warszawa / Warszawie

Lemmatization is the task of determining that two words have the same root, despite their surface differences.

stems :  the central mor- pheme of the word, supplying the main meaning

affixes : adding “additional” meanings of various kinds. 

Stemming : This naive version of morphological analysis

Porter stemmer, a widely used stemming algorithm

source
<pre>
    This was not the map we found in Billy Bones’s chest, but
    an accurate copy, complete in all things-names and heights
    and soundings-with the single exception of the red crosses
    and the written notes.
</pre>

dst
<pre>
    Thi wa not the map we found in Billi Bone s chest but an
    accur copi complet in all thing name and height and sound
    with the singl except of the red cross and the written note
</pre>

https://tartarus.org/martin/PorterStemmer/

## 2.7 Sentence segmentation

based on rules or machine learning

Ex Rules based : Stanford CoreNLP toolkit
https://doi.org/10.3115/v1/P14-5010

## 2.8 Minimum Edit Distance

Given two sequences, an alignment is a correspondence between substrings of the two sequences.

<pre>
I N T E * N T I O N
| | | | | | | | | |
* E X E C U T I O N 
d s s   i s
</pre>
The final row gives the operation list for converting the top string into the bottom string: d for deletion, s for substitution, i for insertion.

We can also assign a particular cost or weight to each of these operations.

Levenshtein distance between two sequences is the simplest weighting factor :
- substitution of a letter for himself : cost=0
- insertion or deletion : cost = 1
- substitution : cost = 2

<pre>
            intention
        del     ins     subst
ntention    intecntion    inxention
</pre>
Finding the edit distance viewed as a search problem.

Obj : find the shorter path to the destination

Dynamic programming is the name for a class of algorithms, first introduced by Bellman (1957), that apply a table-driven method to solve problems by combining solutions to subproblems.

(example of dynamic programming : Viterbi or CKY for parsing)

<pre>
intention 
     │     
┌────┴────┐
│delete i │
└────┬────┘
     ▼     
ntention 
     │     
┌────┴────────┐
│substitute n │
└────┬────────┘
     ▼     
etention 
     │     
┌────┴────────┐
│substitute t │
└────┬────────┘
     ▼       
exention 
     │     
┌────┴────┐
│insert u │
└────┬────┘
     ▼     
exenution 
     │     
┌────┴────────┐
│substitute n │
└────┬────────┘
     ▼     
execution

</pre>

### The minimum edit distance algorithm

D[i, j] = the edit distance between X [1..i] and Y [1.. j], i.e., the first i characters of X and the first j characters of Y .
The edit distance between X and Y is thus D[n, m].

<pre>
               D[i − 1, j] + del-cost(source[i])
D[i, j] = min  D[i, j − 1] + ins-cost(target[ j]) 
               D[i − 1, j − 1] + sub-cost(source[i], target[ j])
</pre>

Using the Levenshtein distance

<pre>
               D[i − 1, j] + 1
D[i, j] = min  D[i, j − 1] + 1 
               D[i − 1, j − 1]  -> 2 if source[i] ̸= target[j] else 0
               
</pre>

Also usefull in alignement, see backtrace.

