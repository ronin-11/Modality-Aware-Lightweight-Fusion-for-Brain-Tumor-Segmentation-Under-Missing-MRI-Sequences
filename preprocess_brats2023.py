import os
import pickle
import numpy as np
import nibabel as nib


# --------------------------------------------------
# BraTS 2023 -> SuperLightNet modality mapping
#
# SuperLightNet channel order:
#   0 = FLAIR
#   1 = T1ce
#   2 = T1
#   3 = T2
# --------------------------------------------------

MODALITIES = ("t2f", "t1c", "t1n", "t2w")

DATA_ROOT = os.path.join(
    "datasets",
    "raw",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)


def load_nifti(filename):
    """Load a NIfTI volume as a NumPy array."""
    return np.asarray(nib.load(filename).dataobj)


def preprocess_subject(subject_dir):
    subject_id = os.path.basename(subject_dir)

    print(f"Processing: {subject_id}")

    # -------------------------
    # Load segmentation
    # -------------------------
    seg_path = os.path.join(
        subject_dir,
        f"{subject_id}-seg.nii.gz"
    )

    label = np.asarray(
        load_nifti(seg_path),
        dtype=np.uint8
    )

    # BraTS 2023 uses ET = 3.
    # Original SuperLightNet expects ET = 4.
    label[label == 3] = 4

    # -------------------------
    # Load the four MRI modalities
    # -------------------------
    images = []

    for modality in MODALITIES:
        image_path = os.path.join(
            subject_dir,
            f"{subject_id}-{modality}.nii.gz"
        )

        image = np.asarray(
            load_nifti(image_path),
            dtype=np.float32
        )

        images.append(image)

    # [W, H, D, 4]
    images = np.stack(images, axis=-1)

    # -------------------------
    # Original SuperLightNet
    # normalization
    # -------------------------
    mask = images.sum(axis=-1) > 0

    for k in range(4):
        x = images[..., k]
        y = x[mask]

        mean = y.mean()
        std = y.std()

        if std > 0:
            x[mask] = (x[mask] - mean) / std

        images[..., k] = x

    # -------------------------
    # Save SuperLightNet-compatible
    # pickle
    # -------------------------
    output_path = os.path.join(
        subject_dir,
        "pkl_ui8f32b0.pkl"
    )

    with open(output_path, "wb") as f:
        pickle.dump((images, label), f)

    print(f"Saved: {output_path}")


def main():
    subjects = sorted(
        [
            os.path.join(DATA_ROOT, name)
            for name in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, name))
        ]
    )

    print(f"Found {len(subjects)} subjects.")

    #for subject_dir in subjects:
        #preprocess_subject(subject_dir)

    preprocess_subject(subjects[0])

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()