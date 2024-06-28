import numpy as np
import torch


class MetricSummary:
    def __init__(self, metric_fn):
        self.metric_fn = metric_fn
        self.__evaluations = []

    def reset(self):
        self.__evaluations = []

    def update(self, input, target):
        evaluation = self.metric_fn(input, target)
        self.__evaluations.append(evaluation)
        return evaluation

    def mean(self):
        if self.__evaluations:
            if isinstance(self.__evaluations[0], torch.Tensor):
                return torch.stack(self.__evaluations).mean()
            else:
                return np.mean(self.__evaluations)
        else:
            raise ValueError('No evaluation exists.')

    def std(self):
        if self.__evaluations:
            if isinstance(self.__evaluations[0], torch.Tensor):
                return torch.stack(self.__evaluations).std()
            else:
                return np.std(self.__evaluations)
        else:
            raise ValueError('No evaluation exists.')

    def count(self):
        return len(self.__evaluations)