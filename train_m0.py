import os
import random
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from brats2023_dataset import BraTS2023Dataset
from resumable_sampler import ResumableBatchSampler
from rng_utils import capture_rng_state, restore_rng_state
from Jnetworks.superlightnet import NormalU_Net
from Jnetworks.JMCriterion import BCEWithDiceLoss


DATA_ROOT = (
    r"datasets/raw/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

TRAIN_SPLIT = r"splits/brats2023/train.txt"
VAL_SPLIT = r"splits/brats2023/val.txt"

CHECKPOINT_DIR = "/content/drive/MyDrive/BrainTumorSegmentation/SuperLightNet/checkpoints/M0_official_seed42"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=40000
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true"
    )

    parser.add_argument(
        "--resume", action="store_true"
    )
    parser.add_argument(
        "--stop-after", type=int, default=None
    )

    args = parser.parse_args()

    # ----------------------------------
    # Reproducibility
    # ----------------------------------

    set_seed(args.seed)

    # ----------------------------------
    # Device
    # ----------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("M0 — SuperLightNet Baseline")
    print("=" * 60)

    print("Device:", device)
    print("Seed:", args.seed)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ----------------------------------
    # Dataset
    # ----------------------------------

    print("\nLoading training dataset...")

    train_dataset = BraTS2023Dataset(
        data_root=DATA_ROOT,
        split_file=TRAIN_SPLIT,
        crop_size=(128, 128, 128),
        training=True,
    )

    print(
        "Training subjects:",
        len(train_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=ResumableBatchSampler(
            dataset_size=len(train_dataset),
            batch_size=args.batch_size,
            seed=args.seed,
        ),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed + 100000),
    )

    # ----------------------------------
    # Model
    # ----------------------------------

    print("\nCreating SuperLightNet...")

    model = NormalU_Net(
        depths_unidirectional="small"
    ).to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", num_parameters)
    print(
        "Parameters (M):",
        num_parameters / 1e6
    )

    # ----------------------------------
    # Loss
    # ----------------------------------

    criterion = BCEWithDiceLoss()

    # ----------------------------------
    # Optimizer
    # ----------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ----------------------------------
    # Cosine learning-rate schedule
    # ----------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_iterations,
    )

    # ----------------------------------
    # Mixed precision
    # ----------------------------------

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    # ----------------------------------
    # Checkpoint directory
    # ----------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # ----------------------------------
    # Training
    # ----------------------------------

    iteration = 0
    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pth")

    if args.resume:
        if not os.path.isfile(latest_path):
            raise FileNotFoundError(latest_path)

        checkpoint = torch.load(
            latest_path,
            map_location=device,
            weights_only=False
        )

        if checkpoint["max_iterations"] != args.max_iterations:
            raise ValueError(
                "Training horizon differs from saved checkpoint."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if "rng_state" not in checkpoint:
            raise ValueError("Checkpoint has no RNG state.")

        if checkpoint.get("seed") != args.seed:
            raise ValueError("Checkpoint seed mismatch.")

        restore_rng_state(checkpoint["rng_state"])

        iteration = checkpoint["iteration"]
        print("Resumed at iteration:", iteration)
        print("RNG states restored.")

    stop_at = (
        args.max_iterations
        if args.stop_after is None
        else min(args.stop_after, args.max_iterations)
    )

    if stop_at < 1:
        raise ValueError("--stop-after must be positive")

    model.train()

    print("\nStarting training...")
    print(
        "Target iterations:",
        1 if args.smoke_test
        else args.max_iterations
    )

    while iteration < stop_at:

        batches_per_epoch = len(train_dataset) // args.batch_size
        if len(train_dataset) % args.batch_size:
            batches_per_epoch += 1

        epoch = iteration // batches_per_epoch
        batch_offset = iteration % batches_per_epoch

        train_loader.batch_sampler.epoch = epoch
        train_loader.batch_sampler.start_batch = batch_offset

        for images, labels, subject_ids in train_loader:

            if iteration >= stop_at:
                break

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # --------------------------
            # Forward + loss
            # --------------------------

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                outputs = model(images)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=False
            ):
                loss_result = criterion(
                    outputs.float(), labels
                )
                loss = (
                    loss_result[0]
                    if isinstance(loss_result, tuple)
                    else loss_result
                )

            # --------------------------
            # Backpropagation
            # --------------------------

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            iteration += 1

            # --------------------------
            # Logging
            # --------------------------

            if (
                iteration == 1
                or iteration % 10 == 0
            ):
                current_lr = (
                    optimizer
                    .param_groups[0]["lr"]
                )

                print(
                    f"Iteration "
                    f"{iteration:05d} | "
                    f"Loss: "
                    f"{loss.item():.6f} | "
                    f"LR: "
                    f"{current_lr:.8f}"
                )

            # --------------------------
            # CPU smoke test
            # --------------------------

            if args.smoke_test:

                print(
                    "\nSMOKE TEST PASSED"
                )

                print(
                    "Input:",
                    images.shape
                )

                print(
                    "Output:",
                    outputs.shape
                )

                print(
                    "Loss:",
                    loss.item()
                )

                return

            # --------------------------
            # Save checkpoint
            # --------------------------

            if iteration % args.save_every == 0:
                checkpoint = {
                    "iteration": iteration,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "seed": args.seed,
                    "max_iterations": args.max_iterations,
                    "rng_state": capture_rng_state(),
                }

                temporary_path = latest_path + ".tmp"
                torch.save(checkpoint, temporary_path)
                os.replace(temporary_path, latest_path)

                print(f"Checkpoint saved at iteration {iteration}")

    # ----------------------------------
    # Final checkpoint
    # ----------------------------------

    print("\nTraining stopped at iteration", iteration)

    if iteration >= args.max_iterations:
        final_path = os.path.join(
            CHECKPOINT_DIR, "m0_final.pth"
        )

        checkpoint = {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "seed": args.seed,
            "max_iterations": args.max_iterations,
            "rng_state": capture_rng_state(),
        }

        temporary_path = final_path + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, final_path)

        print("Final checkpoint saved:", final_path)
    else:
        print("Training paused; no final checkpoint created.")


if __name__ == "__main__":
    main()