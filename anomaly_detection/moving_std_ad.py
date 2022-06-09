from collections import deque
import numpy as np
# import pandas as pd
# from typing import Union, Tuple, Dict, List, Literal

# from .base import AnomalyDetector


# class MovingStdAD(AnomalyDetector):
class MovingStdAD:
    def __init__(
        self, stat_window_size: int, warmup: int = None,
        threshold: float = None, memory_size: int = 5000,
        max_z_score: float = 1.5, min_suspicious: float = 1.2,
        min_anomaly: float = -999999.0, n_suspicious: int = 100
    ):
        """If warmup is None, only threshold will be used
        to determine class."""
        # AnomalyDetector.__init__(self, score_names=None)
        self.stat_window_size = stat_window_size
        assert not (warmup is None and threshold is None),\
            '"warmup" or "threshold" have to be not None.'
        if warmup is not None:
            assert warmup > 2,\
                '2 minimal required warmup steps.'
            assert memory_size > 2,\
                '2 minimal required memory size.'
        self.remaining_warmup = warmup
        self.memory_size = memory_size
        if warmup is None:
            self.memory = None
            self.bin = None
        else:
            self.memory = []
            self.bin = deque([], maxlen=n_suspicious)
        self.max_z_score = max_z_score
        self.min_suspicious = min_suspicious
        self.min_anomaly = min_anomaly
        self.n_suspicious = n_suspicious
        self.threshold = threshold

    def predict_with_stat(
        self, val: float, max_z_score: float = None,
        return_z_score: bool = False
    ) -> bool:
        # TODO: tworzenie rozkładu na podstawie zebranych wspomnień
        is_anomaly = False
        z_score = -1
        if max_z_score is None:
            max_z_score = self.max_z_score
        if val >= self.min_anomaly:
            mean_ = np.mean(self.memory)
            std_ = np.std(self.memory) + 1e-10
            z_score = np.abs(val - mean_) / std_
            is_anomaly = z_score > max_z_score

        if return_z_score:
            return is_anomaly, z_score
        else:
            return is_anomaly

    def predict(
        self, val: float, add_to_memory: bool = True,
        return_z_score: bool = False
    ) -> bool:
        is_anomaly = False
        z_score = -1
        if self.threshold is not None:
            is_anomaly = is_anomaly or self.threshold < val
        if self.memory is not None:
            if self.remaining_warmup <= 0:
                stat_res = self.predict_with_stat(
                    val, max_z_score=self.max_z_score,
                    return_z_score=return_z_score)
                if return_z_score:
                    stat_res, z_score = stat_res
                is_anomaly = is_anomaly or stat_res
            else:
                self.remaining_warmup -= 1

            if add_to_memory:
                self.memory.append(val)
                if len(self.memory) > self.memory_size:
                    self.bin.append(self.memory.pop(0))

        if is_anomaly and self.memory is not None and add_to_memory:
            self.delete_last_suspicious(
                n_suspicious=self.n_suspicious,
                min_suspicious=self.min_suspicious
            )

        if return_z_score:
            return int(is_anomaly), z_score
        else:
            return int(is_anomaly)

    def delete_last_suspicious(self, n_suspicious: int, min_suspicious: float):
        scores = [
            self.predict_with_stat(val)
            for val in self.memory[-n_suspicious:]]

        idx = None
        for i in range(n_suspicious):
            if scores[i] >= min_suspicious:
                idx = i
                break

        # if detected suspicious point, delete it and points following it
        if idx is not None:
            idx = n_suspicious - idx
            recovered = reversed(
                [self.bin.pop() for _ in range(min(idx, len(self.bin)))])
            self.memory = recovered + self.memory[-idx:]
