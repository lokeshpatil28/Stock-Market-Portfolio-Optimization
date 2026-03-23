import numpy as np

class SimplexTableau:

    def __init__(self, T):
        self.T = T
        self.m, self.n = T.shape

    def pivot(self, row, col):
        self.T[row] /= self.T[row, col]
        for i in range(self.m):
            if i != row:
                self.T[i] -= self.T[i, col] * self.T[row]

    def solve(self):
        while True:
            cost = self.T[-1, :-1]
            if np.all(cost >= -1e-9):
                break

            col = np.argmin(cost)

            ratios = []
            for i in range(self.m-1):
                if self.T[i, col] > 1e-9:
                    ratios.append(self.T[i, -1] / self.T[i, col])
                else:
                    ratios.append(np.inf)

            row = np.argmin(ratios)

            self.pivot(row, col)

        return self.T