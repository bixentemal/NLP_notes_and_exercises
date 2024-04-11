import unittest

# Write a program to compute unsmoothed unigrams and bigrams.
# Run your n-gram program on two different small corpora of your choice (you might use email text or newsgroups). Now compare the statistics of the two corpora. What are the differences in the most common unigrams between the two? How about interesting differences in bigrams?
# Add an option to your program to generate random sentences.
# Add an option to your program to compute the perplexity of a test set.
class MyTestCase(unittest.TestCase):

    def compute_unigram(self, corpus: str):
        return None

    def compute_bigram(self, corpus: str):
        return None

    def generate_random_sentences(self, ngram):
        return None

    def compute_perplexity(self, ngram):
        return None

    def test(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
