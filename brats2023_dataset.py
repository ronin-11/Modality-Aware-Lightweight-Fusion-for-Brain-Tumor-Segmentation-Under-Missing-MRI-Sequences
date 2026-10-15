import os
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# SuperLightNet channel order:
# 0 = FLAIR, 1 = T1ce, 2 = T1, 3 = T2
MODALITIES = ("t2f", "t1c", "t1n", "t2w")


class BraTS2023Dataset(Dataset):

    def __init__(
        self,
        data_root,
        split_file,
        crop_size=(128, 128, 128),
        training=False,
    ):
        self.data_root = data_root
        self.crop_size = crop_size
        self.training = training

        with open(split_file, "r") as f:
            self.subjects = [
                line.strip()
                for line in f
                if line.strip()
            ]

    def __len__(self):
        return len(self.subjects)

    def _load_volume(self, path):
        return np.asarray(
            nib.load(path).dataobj,
            dtype=np.float32
        )

    def _center_crop(self, image, label):
        """Center crop image and label to crop_size."""

        sx, sy, sz = self.crop_size
        x, y, z = label.shape

        x0 = (x - sx) // 2
        y0 = (y - sy) // 2
        z0 = (z - sz) // 2

        image = image[
            x0:x0 + sx,
            y0:y0 + sy,
            z0:z0 + sz,
            :
        ]

        label = label[
            x0:x0 + sx,
            y0:y0 + sy,
            z0:z0 + sz
        ]

        return image, label

    def _random_flip(self, image, label):

        for axis in range(3):
            if random.random() < 0.5:
                image = np.flip(image, axis=axis)
                label = np.flip(label, axis=axis)

        return image, label

    def _intensity_shift(self, image):

        scale = np.random.uniform(0.9, 1.1, size=(1, 1, 1, 4))
        shift = np.random.uniform(-0.1, 0.1, size=(1, 1, 1, 4))

        image = image * scale + shift

        return image

    def __getitem__(self, index):

        subject_id = self.subjects[index]
        subject_dir = os.path.join(
            self.data_root,
            subject_id
        )

        # ----------------------------------
        # Load modalities
        # ----------------------------------

        modalities = []

        for modality in MODALITIES:

            path = os.path.join(
                subject_dir,
                f"{subject_id}-{modality}.nii.gz"
            )

            volume = self._load_volume(path)
            modalities.append(volume)

        image = np.stack(modalities, axis=-1)

        # ----------------------------------
        # Load segmentation
        # ----------------------------------

        seg_path = os.path.join(
            subject_dir,
            f"{subject_id}-seg.nii.gz"
        )

        label = np.asarray(
            nib.load(seg_path).dataobj,
            dtype=np.uint8
        )

        # BraTS 2023 ET=3 -> SuperLightNet ET=4
        label[label == 3] = 4

        # ----------------------------------
        # Original-style normalization
        # ----------------------------------

        mask = image.sum(axis=-1) > 0

        for k in range(4):

            x = image[..., k]
            values = x[mask]

            mean = values.mean()
            std = values.std()

            if std > 0:
                x[mask] = (x[mask] - mean) / std

            image[..., k] = x

        # ----------------------------------
        # 128³ crop
        # ----------------------------------

        image, label = self._center_crop(
            image,
            label
        )

        # ----------------------------------
        # Training augmentation
        # ----------------------------------

        if self.training:

            image, label = self._random_flip(
                image,
                label
            )

            image = self._intensity_shift(image)

        # np.flip can create negative strides,
        # therefore force contiguous arrays.
        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)

        # [H,W,D,4] -> [4,H,W,D]
        image = torch.from_numpy(
            image.transpose(3, 0, 1, 2)
        ).float()

        label = torch.from_numpy(label).long()

        return image, label, subject_id