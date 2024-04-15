import unittest
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import random


# Write a program to compute unsmoothed unigrams and bigrams.
# Run your n-gram program on two different small corpora of your choice (you might use email text or newsgroups). Now compare the statistics of the two corpora. What are the differences in the most common unigrams between the two? How about interesting differences in bigrams?
# Add an option to your program to generate random sentences.
# Add an option to your program to compute the perplexity of a test set.
class MyTestCase(unittest.TestCase):

    def compute_unigram(self, corpus) -> Dict[str, int]:
        result = defaultdict(int)
        for word in corpus.split():
            result[word] += 1
        return result

    def grouped(self, iterable, n):
        return zip(*[iter(iterable)] * n)

    def sample_sentence(self, n_gram_list: List[Tuple[Any, float]], sampling_len = 20) -> str:
        result = ""
        tlist = n_gram_list
        tlist.sort(key=lambda x: x[1], reverse = True)
        scores_col = [x[1] for x in tlist]
        #np_scores_col = np.array(scores_col)
        #np_scores_col_sm = softmax(np_scores_col)
        for round in range(sampling_len):
            r = random.uniform(min(scores_col), max(scores_col))
            match_gram = tlist[-1]
            #for a,b in self.grouped(tlist, 2):
            for a in tlist:
                if a[1] < r :
                    match_gram = a[0]
                    break

            result += " " + match_gram
        return result

    def compute_bigram(self, corpus: str):
        return None

    def compute_perplexity(self, ngram):
        return None

    def test_unigram(self):
        lines = []
        # from https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories-valid.txt
        with open("corpus.txt") as f:
            lines = f.readlines()
        lines = map(lambda x: "<s>" + x[:-1] + "</s>", lines)
        lines = map(lambda x: x.replace(":", " "), lines)
        lines = map(lambda x: x.replace(",", " "), lines)
        lines = map(lambda x: x.replace("\"", " "), lines)
        lines = map(lambda x: x.replace("'", " "), lines)
        lines = map(lambda x: ' '.join(x.split()), lines)
        lines = map(lambda x: x.replace(".", "</s><s>"), lines)
        lines = map(lambda x: x.replace("!", "</s><s>"), lines)
        lines = map(lambda x: x.replace("<s> ", "<s>"), lines)
        lines = map(lambda x: x.replace(" </s>", "</s>"), lines)
        lines = map(lambda x: x.replace("<s></s>", ""), lines)
        lines = map(lambda x: x.replace("<s>", " <s> "), lines)
        lines = map(lambda x: x.replace("</s>", " </s> "), lines)
        #remove start and end sentence
        lines = map(lambda x: x.replace("</s>", ""), lines)
        lines = map(lambda x: x.replace("<s>", ""), lines)
        flatten = ""
        for l in lines:
            flatten = flatten + l
        d = self.compute_unigram(flatten)
        vocab_size = len(d)
        unigrams = list(map(lambda x : (x[0], float(float(x[1])/vocab_size)), list(d.items())))
        print(self.sample_sentence(unigrams, 20))
        #self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
