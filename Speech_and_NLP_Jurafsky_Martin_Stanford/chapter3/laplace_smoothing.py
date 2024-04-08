import unittest
from tabulate import tabulate


class MyTestCase(unittest.TestCase):
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

    def mle(self, count_w: int, total_count_w_1: int) -> float:
        return count_w / total_count_w_1

    def laplace(self, count_w: int, total_count_w_1: int) -> float:
        return (count_w+1) / (total_count_w_1+self.vocabulary_size)

    def reconstructed_count(self, count_w: int, total_count_w_1: int) -> float:
        return (count_w+1)*total_count_w_1 / (total_count_w_1+self.vocabulary_size)

    def apply(self, func):
        print(tabulate(self.berkeley_sample))
        result = []
        for row in self.berkeley_sample:
            result_row = []
            row_index = 0;
            for i in row:
                result_row.append(func(i, self.berkeley_total[row_index]))
            result.append(result_row)
            row_index += 1
        print(tabulate(result))

    def test_bigram_mle(self):
        self.apply(self.mle)

    def test_laplace_smoothing(self):
        self.apply(self.laplace)

    def test_reconstructed_count(self):
        self.apply(self.reconstructed_count)

if __name__ == '__main__':
    unittest.main()
