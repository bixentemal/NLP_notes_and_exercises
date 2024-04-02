import unittest


class ExercisesChapter2(unittest.TestCase):

    def min_edit_distance(self, source:str, target:str) -> int:
        def subst_cost(x, y):
            return 0 if x == y else 2
        def del_cost(x):
            return 1
        def insert_cost(x):
            return 1

        distance_matrix = [ [None]*(len(target)+1) for i in range(len(source)+1)]
        distance_matrix[0][0] = 0

        # each row init
        for i in range(1, len(source)+1):
            distance_matrix[i][0] = distance_matrix[i-1][0] + del_cost(source[i-1])

        for j in range(1, len(target)+1):
            distance_matrix[0][j] = distance_matrix[0][j-1] + insert_cost(target[j-1])

        for i in range(1, len(source)+1):
            for j in range(1, len(target) + 1):
                distance_matrix[i][j] = min(
                    distance_matrix[i-1][j] + del_cost(source[i-1]),
                    distance_matrix[i - 1][j-1] + subst_cost(source[i-1], target[j-1]),
                    distance_matrix[i][j-1] + insert_cost(target[j-1]))

        return distance_matrix[len(source)][len(target)];

    def test_count_distance24(self):
        # ex 2.4
        w1 = "leda"
        w2 = "deal"
        self.assertEqual(self.min_edit_distance(w1,w2), 4)  # add assertion here

    def test_count_distance_ref(self):
        w1 = "intention"
        w2 = "execution"
        self.assertEqual(self.min_edit_distance(w1,w2), 8)  # add assertion here

    def test_count_distance25(self):
        # ex 2.5
        w1 = "drive"
        w2 = "divers"
        w3 = "brief"
        dw2 = self.min_edit_distance(w1, w2)
        dw3 = self.min_edit_distance(w1, w3)
        if dw2 > dw3 :
            print("%s(%d) is closer to %s than %s(%d)"%(w3, dw3, w1, w2, dw2))
        else:
            print("%s(%d) is closer to %s than %s(%d)"%(w2, dw2, w1, w3, dw3))

if __name__ == '__main__':
    unittest.main()
