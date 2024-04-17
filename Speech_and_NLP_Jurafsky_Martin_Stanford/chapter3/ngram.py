import math
from collections import defaultdict
from typing import List, Tuple, Any
import random

def load_as_single_str(file_in: str) -> str:
    result = None
    with open(file_in) as f:
        result = f.read()
    return result


sos_tok = "<s>"
eos_tok = "</s>"


def tokenize(content: str) -> List[str]:
    result_str = content.lower()

    sentence_separator = [".", "!", ";", ":"]

    for sep in sentence_separator:
        result_str = result_str.replace(sep, eos_tok + sos_tok)

    exclude_symbol = ["\"", "'", ","]
    for s in exclude_symbol:
        result_str = result_str.replace(s, " ")

    result_str = result_str.replace("<s> ", "<s>")
    result_str = result_str.replace(" </s>", "</s>")
    result_str = result_str.replace("<s></s>", "")
    result_str = result_str.replace("<s>", " <s> ")
    result_str = result_str.replace("</s>", " </s> ")

    # transform a corpus into a list of words
    return result_str.split()


def grouped(iterable, n):
    if n == 1:
        return [sos_tok] + list(iterable) + [eos_tok]
    elif n == 2:
        l1 = ["<s>" for n in range(1)] + list(iterable)
        l2 = list(iterable) + ["</s>" for n in range(1)]
        # return zip(*[iter(iterable)] * n)
        return list(zip(l1, l2))
    else:
        raise Exception("Only unigram and bigram supported")


def count_stats(tokens: List[str], n=1) -> List[Tuple[Any, int]]:
    # returns the count stats as a list of tuple (token, count)
    # ordered alphabetically by token
    result_dict = defaultdict(int)
    for word in grouped(tokens, n):
        result_dict[word] += 1

    res = [(k, v) for k, v in result_dict.items()]
    res.sort()
    return res


# def flat(i_matrix: List[List[float|long]]) -> List[float|long]:
#	pass

# def matrixize(i_list: List[float|long]) -> List[List[float|long]]:
#	pass

def matrixize(ngrams: List[Tuple[Any, int]], n=1) -> List[List[float | int]]:
    if n == 1:
        # one nested array only
        ngram_sorted = ngrams
        ngram_sorted.sort()
        return [[count for _, count in ngram_sorted]]
    elif n == 2:
        # get as 'vocabulary' unigram
        unigrams = set()
        for t, count in ngrams:
            unigrams.add(t[0])
            unigrams.add(t[1])
        unigrams_list = list(unigrams)
        unigrams_list.sort()
        #result = [[0]*len(unigrams_list)]*len(unigrams_list)
        result = []
        for i in range(len(unigrams_list)):
            result.append([0]*len(unigrams_list))
        for t, count in ngrams:
            result[unigrams_list.index(t[0])][unigrams_list.index(t[1])] = count
        return result
    else:
        raise Exception("Only unigram and bigram supported")


def apply_transformation(
        i_matrix: List[List[float | int]],
        i_total: List[int],
        vocabulary_size:int,
        transfo_func) -> List[List[float | int]]:
    result = []
    for row in i_matrix:
        result_row = []
        row_index = 0
        for i in row:
            result_row.append(transfo_func(i, i_total[row_index], vocabulary_size))
        result.append(result_row)
    return result


def mle(count_w: int, total_count_w_1: int, vocabulary_size: int) -> float:
    return count_w / total_count_w_1


def laplace(count_w: int, total_count_w_1: int, vocabulary_size: int) -> float:
    return (count_w + 1) / (total_count_w_1 + vocabulary_size)


def reconstructed_count(self, count_w: int, total_count_w_1: int, vocabulary_size: int) -> float:
    return (count_w + 1) * total_count_w_1 / (total_count_w_1 + vocabulary_size)


def sample_sentence(prompt: str, ngram_as_matrix, vocabulary: List[str], max_len=20, n=1) -> str:
    sentence_idx = sample_sentence_as_vocalbulary_index(vocabulary.index(prompt), ngram_as_matrix, max_len=max_len, n=n)
    result = " ".join([vocabulary[idx] for idx in sentence_idx])
    return result

def sample_sentence_as_vocalbulary_index_unigram(prompt_idx: int, unigram_list: List[float], max_len=20, n=1) -> List[int]:
    result = [prompt_idx]
    score_list = unigram_list
    for round in range(max_len-1):
        r = random.uniform(min(score_list), max(score_list))
        match_gram_idx = len(score_list)-1
        count = 0
        best_dist = 100000000
        for a in score_list:
            curr_dist = abs(a-r)
            if curr_dist < best_dist:
                match_gram_idx = count
                best_dist = curr_dist
            count += 1
        result.append(match_gram_idx)
    return result

def sample_sentence_as_vocalbulary_index_bigram(prompt_idx: int, ngram_as_matrix, max_len=20, n=1) -> List[int]:
    result = [prompt_idx]
    for round in range(max_len - 1):
        score_list = ngram_as_matrix[result[-1]]
        r = random.uniform(min(score_list), max(score_list))
        match_gram_idx = len(score_list) - 1
        count = 0
        best_dist = 100000000
        for a in score_list:
            curr_dist = abs(a - r)
            if curr_dist < best_dist:
                match_gram_idx = count
                best_dist = curr_dist
            count += 1
        result.append(match_gram_idx)
    return result

def sample_sentence_as_vocalbulary_index(prompt_idx: int, ngram_as_matrix, max_len=20, n=1) -> List[int]:
    if n == 1:
        return sample_sentence_as_vocalbulary_index_unigram(prompt_idx, ngram_as_matrix, max_len=max_len, n=n)
    elif n == 2:
        return sample_sentence_as_vocalbulary_index_bigram(prompt_idx, ngram_as_matrix, max_len=max_len, n=n)
    else:
        raise Exception("Only unigram and bigram supported")

def perplexity(ngram_as_matrix, vocabulary: List[str], sentence: str, n=1) -> float:
    tokens = sentence.split()
    perplexity = 1
    if n == 1:
        for t in tokens:
            # get probability for current token
            p = ngram_as_matrix[vocabulary.index(t)]
            #print(p)
            perplexity = perplexity * 1/p
        perplexity = perplexity ** (1. / len(tokens))
    elif n == 2:
        tokens1 = [None] + tokens
        tokens2 = tokens + [None]
        for t in list(zip(tokens1, tokens2))[1:-1]:
            # get probability for current token
            p = ngram_as_matrix[vocabulary.index(t[0])][vocabulary.index(t[1])]
            #print(p)
            perplexity = perplexity * 1 / p
        perplexity = perplexity ** (1. / len(tokens))
    else:
        raise Exception("Only unigram and bigram supported")
    return perplexity


