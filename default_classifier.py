import math
import numpy as np
import pandas as pd
import training_result
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

    def update_weight(
        self,
        lr: float,
        weight: NDArray[np.float64],
        y: float,
        y_hat: float,
        x: NDArray[np.float64],
    ):
        base = self.delta_bias(y, y_hat) * x
        new_weight = weight - (lr * base)
        return new_weight

    def update_bias(self, lr: float, bias: float, y: float, y_hat: float):
        base = self.delta_bias(y, y_hat)
        new_bias = bias - (lr * base)
        return new_bias

    def delta_bias(self, y: float, y_hat: float):
        return 2 * (y_hat - y) * (1 - y_hat) * y_hat

    def squared_error(self, y: float, y_hat: float):
        return math.pow((y - y_hat), 2)

    def fit(
        self,
        datasets: pd.DataFrame,
        learning_rate: float,
        init_weight: float,
        init_bias: float,
        feature_columns: list,
        target_column: str,
    ):
        theta = []
        i = 0
        while i < len(feature_columns):
            theta.append(init_weight)
            i += 1
        bias = init_bias
        weight = np.array(theta)

        squared_errors = []
        verdicts = []
        total_index, _ = datasets.shape
        for i, (_, row) in enumerate(datasets.iterrows()):
            x = []
            for col in feature_columns:
                x.append(row[col])
            x = np.array(x)

            h = self.h(weight=weight, x=x, bias=bias)
            sigmoid = self.sigmoid(h)
            y_label = float(row[target_column])  # type: ignore

            squared_error = self.squared_error(y_label, sigmoid)
            squared_errors.append(squared_error)

            predict = self.predict(sigmoid)
            verdict = predict == y_label
            verdicts.append(verdict)

            if i < total_index - 1:
                weight = self.update_weight(learning_rate, weight, y_label, sigmoid, x)
                bias = self.update_bias(learning_rate, bias, y_label, sigmoid)

        result = training_result.TrainingResult(
            n=total_index,
            weight=weight,
            bias=bias,
            squared_error=np.array(squared_errors),
            verdict=np.array(verdicts),
        )
        return result
