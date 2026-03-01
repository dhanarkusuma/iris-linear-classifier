from numpy.typing import NDArray


class EpochResult:
    def __init__(
        self,
        weight: NDArray,
        bias: float,
        mse_per_epoch: list,
        accuracy_per_epoch: list,
    ):
        self.weight = weight
        self.bias = bias
        self.mse_per_epoch = mse_per_epoch
        self.accuracy_per_epoch = accuracy_per_epoch

    def GetWeight(self):
        return self.weight

    def GetBias(self):
        return self.bias

    def GetMSEPerEpoch(self):
        return self.mse_per_epoch

    def GetAccuracyPerEpoch(self):
        return self.accuracy_per_epoch
