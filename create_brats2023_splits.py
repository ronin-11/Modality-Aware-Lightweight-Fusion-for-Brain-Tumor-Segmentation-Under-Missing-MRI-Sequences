import os
import random

DATA_ROOT = os.path.join(
    "datasets",
    "raw",
    "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

OUTPUT_DIR = os.path.join("splits", "brats2023")

SEED = 42


def main():

    subjects = sorted([
        name for name in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, name))
        and name.startswith("BraTS-GLI-")
    ])

    print(f"Total subjects found: {len(subjects)}")

    # Fixed deterministic shuffle
    rng = random.Random(SEED)
    rng.shuffle(subjects)

    # Fixed 70/10/20 split for 1251 subjects
    n_train = int(0.70 * len(subjects))
    n_val = int(0.10 * len(subjects))

    train = subjects[:n_train]
    val = subjects[n_train:n_train + n_val]
    test = subjects[n_train + n_val:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    splits = {
        "train.txt": train,
        "val.txt": val,
        "test.txt": test,
    }

    for filename, cases in splits.items():
        path = os.path.join(OUTPUT_DIR, filename)

        with open(path, "w") as f:
            for case in cases:
                f.write(case + "\n")

    print("\nSplit complete")
    print("--------------------")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Total: {len(train) + len(val) + len(test)}")
    print(f"Seed:  {SEED}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()