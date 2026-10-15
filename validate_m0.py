
import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from brats2023_dataset import BraTS2023Dataset
from Jnetworks.superlightnet import NormalU_Net
from evaluation_metrics import evaluate_brats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    project = Path(__file__).resolve().parent

    drive = Path(
        "/content/drive/MyDrive/BrainTumorSegmentation/SuperLightNet"
    )

    data_root = (
        project / "datasets/raw/"
        "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    )

    split_file = project / "splits/brats2023/val.txt"

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else drive / "checkpoints/M0_official_seed42/latest.pth"
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else drive / "results/M0"
    )

    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_root}"
        )

    if not split_file.exists():
        raise FileNotFoundError(
            f"Validation split not found: {split_file}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Evaluation: 128^3 center crop")

    dataset = BraTS2023Dataset(
        data_root=str(data_root),
        split_file=str(split_file),
        crop_size=(128, 128, 128),
        training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = NormalU_Net(
        depths_unidirectional="small"
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    iteration = checkpoint["iteration"]

    print("Loaded iteration:", iteration)
    print("Validation subjects:", len(dataset))

    rows = []

    with torch.inference_mode():
        for images, labels, subject_ids in loader:
            images = images.to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)
            predictions = predictions.long()
            predictions[predictions == 3] = 4

            predictions = predictions.cpu().numpy()
            labels = labels.numpy()

            for i, subject_id in enumerate(subject_ids):
                flair_path = (
                    data_root / subject_id /
                    f"{subject_id}-t2f.nii.gz"
                )

                seg_path = (
                    data_root / subject_id /
                    f"{subject_id}-seg.nii.gz"
                )

                flair = nib.load(str(flair_path))
                seg = nib.load(str(seg_path))

                if flair.shape != seg.shape:
                    raise ValueError(
                        f"Shape mismatch: {subject_id}"
                    )

                if not np.allclose(
                    flair.affine, seg.affine, atol=1e-5
                ):
                    raise ValueError(
                        f"Affine mismatch: {subject_id}"
                    )

                spacing = tuple(
                    float(x)
                    for x in seg.header.get_zooms()[:3]
                )

                scores = evaluate_brats(
                    predictions[i],
                    labels[i],
                    spacing=spacing,
                )

                row = {
                    "subject_id": subject_id,
                    "iteration": iteration,
                    "evaluation": "center_crop_128",
                    **scores,
                }

                rows.append(row)

                print(
                    f"{subject_id} | "
                    f"ET Dice={scores['ET_Dice']:.4f} | "
                    f"TC Dice={scores['TC_Dice']:.4f} | "
                    f"WT Dice={scores['WT_Dice']:.4f}"
                )

                if args.limit > 0 and len(rows) >= args.limit:
                    break

            if args.limit > 0 and len(rows) >= args.limit:
                break

    if not rows:
        raise RuntimeError("No subjects evaluated.")

    print("\nAVERAGE METRICS")

    metric_names = [
        "ET_Dice", "TC_Dice", "WT_Dice",
        "ET_HD95", "TC_HD95", "WT_HD95",
    ]

    for name in metric_names:
        values = np.array(
            [row[name] for row in rows],
            dtype=float,
        )

        valid = values[np.isfinite(values)]

        mean = (
            float(np.mean(valid))
            if len(valid)
            else float("nan")
        )

        print(
            f"{name}: {mean:.4f} "
            f"(valid={len(valid)}/{len(values)})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir /
        f"validation_iteration_{iteration}_n{len(rows)}_dice_hd95.csv"
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "iteration",
                "evaluation",
                *metric_names,
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved:", output_path)


if __name__ == "__main__":
    main()
