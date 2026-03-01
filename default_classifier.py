import math
import numpy as np
import pandas as pd
from numpy.typing import NDArray


class DefaultClassifier:
    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def h(self, weight: NDArray[np.float64], x: NDArray[np.float64], bias: float):
        return weight.T @ x + bias

    def predict(self, y_hat):
        if y_hat > 0.5:
            return 1
        else:
            return 0

    def update_weight(self, y: float, y_hat: float, x: NDArray[np.float64]):
        base = self.update_bias(y, y_hat)
        delta = x * base
        return delta

    def update_bias(self, y: float, y_hat: float):
        return 2 * (y_hat - y) * (1 - y_hat) * y_hat

    def squared_error(self, y: float, y_hat: float):
        return math.pow((y - y_hat), 2)

    def fit(
        self,
        datasets: pd.DataFrame,
        init_weight: float,
        init_bias: float,
        feature_columns: list,
    ):
        theta = []
        i = 0
        while i < len(feature_columns):
            theta.append(init_weight)
            i += 1
        bias = init_bias
        weight = np.array(theta)

        for index, row in datasets.iterrows():
            x = []
            for col in feature_columns:
                x.append(row[col])
            x = np.array(x)

            h = self.h(weight=weight, x=x, bias=bias)
            sigmoid = self.sigmoid(h)
            predict = self.predict(sigmoid)

        return 0
