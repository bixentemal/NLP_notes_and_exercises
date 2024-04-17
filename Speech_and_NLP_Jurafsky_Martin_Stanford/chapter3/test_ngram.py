import unittest

from chapter3.ngram import load_as_single_str, tokenize, count_stats, matrixize, apply_transformation, laplace, \
    sample_sentence, perplexity


class MyTestCase(unittest.TestCase):
    def test_load_as_single_str(self):
        content = load_as_single_str("corpus.txt")
        self.assertEqual(len(content), 15863)

    def test_tokenize(self):
        tokens_with_seos = tokenize(load_as_single_str("corpus.txt"))
        self.assertEqual(len(tokens_with_seos), 3926)

    def test_count_stats(self):
        tokens_with_seos = tokenize(load_as_single_str("corpus.txt"))[0:30]
        unigrams = count_stats(tokens_with_seos, n=1)
        bigrams = count_stats(tokens_with_seos, n=2)
        self.assertEqual(len(unigrams), 20)
        self.assertEqual(len(bigrams), 27)

    def test_matrixize(self):
        unigrams = [("a", 2), ("c", 1), ("b", 5)]
        unigrams_as_m = matrixize(unigrams)
        self.assertEqual(unigrams_as_m, [[2,5,1]])
        bigrams = [(("a", "b"), 2), (("b", "c"), 1), (("b", "a"), 5)]
        bigrams_as_m = matrixize(bigrams, n=2)
        self.assertEqual(bigrams_as_m, [[0, 2, 0], [5, 0, 1], [0, 0, 0]])

    def test_apply_transformation(self):
        berkeley_total = [2533, 927, 2417, 746, 158, 1093, 341, 278]
        berkeley_sample = \
            [[5, 827, 0, 9, 0, 0, 0, 2],
             [2, 0, 608, 1, 6, 6, 5, 1],
             [2, 0, 4, 686, 2, 0, 6, 211],
             [0, 0, 2, 0, 16, 2, 42, 0],
             [1, 0, 0, 0, 0, 82, 1, 0],
             [15, 0, 15, 0, 1, 4, 0, 0],
             [2, 0, 0, 0, 0, 1, 0, 0],
             [1, 0, 1, 0, 0, 0, 0, 0]]
        vocabulary_size = 1446
        laplace_smoothing = apply_transformation(berkeley_sample, berkeley_total, vocabulary_size, laplace)
        self.assertEqual([[0.0015079165619502387, 0.20809248554913296, 0.00025131942699170643, 0.0025131942699170647,
                           0.00025131942699170643, 0.00025131942699170643, 0.00025131942699170643,
                           0.0007539582809751194],
                          [0.0007539582809751194, 0.00025131942699170643, 0.15305353103794922, 0.0005026388539834129,
                           0.0017592359889419453, 0.0017592359889419453, 0.0015079165619502387, 0.0005026388539834129],
                          [0.0007539582809751194, 0.00025131942699170643, 0.0012565971349585323, 0.17265644634330235,
                           0.0007539582809751194, 0.00025131942699170643, 0.0017592359889419453, 0.05327971852224177],
                          [0.00025131942699170643, 0.00025131942699170643, 0.0007539582809751194,
                           0.00025131942699170643, 0.0042724302588590096, 0.0007539582809751194, 0.010806735360643378,
                           0.00025131942699170643],
                          [0.0005026388539834129, 0.00025131942699170643, 0.00025131942699170643,
                           0.00025131942699170643, 0.00025131942699170643, 0.020859512440311635, 0.0005026388539834129,
                           0.00025131942699170643],
                          [0.004021110831867303, 0.00025131942699170643, 0.004021110831867303, 0.00025131942699170643,
                           0.0005026388539834129, 0.0012565971349585323, 0.00025131942699170643,
                           0.00025131942699170643],
                          [0.0007539582809751194, 0.00025131942699170643, 0.00025131942699170643,
                           0.00025131942699170643, 0.00025131942699170643, 0.0005026388539834129,
                           0.00025131942699170643, 0.00025131942699170643],
                          [0.0005026388539834129, 0.00025131942699170643, 0.0005026388539834129, 0.00025131942699170643,
                           0.00025131942699170643, 0.00025131942699170643, 0.00025131942699170643,
                           0.00025131942699170643]], laplace_smoothing)

    def test_apply_transformation2(self):
        tokens_with_seos = tokenize(load_as_single_str("corpus.txt"))
        unigrams = count_stats(tokens_with_seos, n=1)
        unigrams_as_m = matrixize(unigrams)
        total = unigrams_as_m[0]
        vocabulary_size = len(total)

        bigrams = count_stats(tokens_with_seos, n=2)
        unigrams_as_m = matrixize(bigrams, n=2)

        # resize to 10x10 for test
        unigrams_as_m_resized = []
        #wlist = [v for v,_ in unigrams[0:10]]
        #unigrams_as_m_resized.append([" "] + wlist)
        for i in range(10):
            #unigrams_as_m_resized.append([wlist[i]] + unigrams_as_m[i][0:10])
            unigrams_as_m_resized.append(unigrams_as_m[i][0:10])
        #print(tabulate(unigrams_as_m_resized))
        laplace_smoothing = apply_transformation(unigrams_as_m_resized, total, vocabulary_size, laplace)
        #print(tabulate(laplace_smoothing))
        #print(laplace_smoothing)
        self.assertEqual([[0.0011312217194570137, 0.3902714932126697, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.0022624434389140274, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.004524886877828055, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0022624434389140274, 0.0011312217194570137],
                          [0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.0011312217194570137, 0.0011312217194570137, 0.0022624434389140274, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.003393665158371041, 0.0011312217194570137, 0.0022624434389140274, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.005656108597285068, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.0022624434389140274, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.003393665158371041, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137],
                          [0.0022624434389140274, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137, 0.0011312217194570137,
                           0.0011312217194570137, 0.0011312217194570137]], laplace_smoothing)

    def test_sample_sentence(self):
        tokens_with_seos = tokenize(load_as_single_str("corpus.txt"))
        unigrams = count_stats(tokens_with_seos, n=1)
        unigrams_as_m = matrixize(unigrams, n=1)
        vocab = [v for v, _ in unigrams]
        unigrams_proba = list(map(lambda x: float(float(x) / len(tokens_with_seos)), unigrams_as_m[0]))
        res = sample_sentence("the", unigrams_proba, vocab, n=1)
        print(res)
        #self.assertEqual(True, False)  # add assertion here

    def test_sample_sentence_bigram(self):
        tokens_with_seos = tokenize(load_as_single_str("corpus.txt"))
        unigrams = count_stats(tokens_with_seos, n=1)
        unigrams_as_m = matrixize(unigrams)
        total = unigrams_as_m[0]
        vocab = [v for v, _ in unigrams]
        bigrams = count_stats(tokens_with_seos, n=2)
        bigrams_as_m = matrixize(bigrams, n=2)
        laplace_smoothing = apply_transformation(bigrams_as_m, total, len(vocab), laplace)
        res = sample_sentence("the", laplace_smoothing, vocab, n=2)
        print(res)

    def test_perplexity(self):
        tokens_with_seos = tokenize(load_as_single_str("corpus.txt"))
        unigrams = count_stats(tokens_with_seos, n=1)
        unigrams_as_m = matrixize(unigrams, n=1)
        total = unigrams_as_m[0]
        vocab = [v for v, _ in unigrams]
        unigrams_model = list(map(lambda x: float(float(x) / len(tokens_with_seos)), unigrams_as_m[0]))
        bigrams = count_stats(tokens_with_seos, n=2)
        bigrams_as_m = matrixize(bigrams, n=2)
        bigram_model_smoothed = apply_transformation(bigrams_as_m, total, len(vocab), laplace)
        #perplexity_sample = "<s> once upon a time there was a little boy named tim tim had a big orange ball  one day tim met a girl named sue </s> <s> sue had a pretty doll </s> <s> tim liked sue s doll and sue liked tim s orange ball </s> <s> tim and sue thought about a trade </s> <s> they would trade the ball for the doll </s> <s> tim was not sure </s> "
        perplexity_sample = " ".join(tokens_with_seos[10:50])
        punigram = perplexity(unigrams_model, vocab, perplexity_sample, n=1)
        print("Perplexity unigram = \t\t\t"+str(punigram))
        pbigram = perplexity(bigram_model_smoothed, vocab, perplexity_sample, n=2)
        print("Perplexity (smoothed) bigram = \t"+str(pbigram))




if __name__ == '__main__':
    unittest.main()
