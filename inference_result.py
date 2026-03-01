from numpy.typing import NDArray


class InferenceResult:
    def __init__(
        self,
        n: int,
        squared_error: NDArray,
        verdict: NDArray,
    ):
        self.squared_error = squared_error
        self.n = n
        self.verdict = verdict

    def MSE(self):
        return self.squared_error.sum() / self.n

    def Accuracy(self):
        return self.verdict.sum() / self.n
