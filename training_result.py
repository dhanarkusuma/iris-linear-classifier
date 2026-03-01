from numpy.typing import NDArray


class TrainingResult:
    def __init__(
        self,
        n: int,
        weight: NDArray,
        bias: float,
        squared_error: NDArray,
        verdict: NDArray,
    ):
        self.weight = weight
        self.bias = bias
        self.squared_error = squared_error
        self.n = n
        self.verdict = verdict

    def MSE(self):
        return self.squared_error.sum() / self.n

    def Accuracy(self):
        return self.verdict.sum() / self.n
