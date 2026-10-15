
import math

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler):
    """
    Deterministic epoch-wise shuffled batches.

    start_batch is the batch offset within the current epoch.
    """

    def __init__(
        self,
        dataset_size,
        batch_size=1,
        seed=42,
        epoch=0,
        start_batch=0,
    ):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.start_batch = start_batch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = torch.randperm(
            self.dataset_size,
            generator=generator,
        ).tolist()

        batches = [
            indices[i:i + self.batch_size]
            for i in range(
                0,
                self.dataset_size,
                self.batch_size,
            )
        ]

        yield from batches[self.start_batch:]

    def __len__(self):
        total_batches = math.ceil(
            self.dataset_size / self.batch_size
        )
        return max(0, total_batches - self.start_batch)
