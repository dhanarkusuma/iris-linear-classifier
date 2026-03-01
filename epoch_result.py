from numpy.typing import NDArray


class EpochResult:
    def __init__(
        self,
        weight: NDArray,
        bias: float,
        mse_last_epoch: float,
        accuracy_last_epoch: float,
    ):
        self.weight = weight
        self.bias = bias
        self.mse_last_epoch = mse_last_epoch
        self.accuracy_last_epoch = accuracy_last_epoch
