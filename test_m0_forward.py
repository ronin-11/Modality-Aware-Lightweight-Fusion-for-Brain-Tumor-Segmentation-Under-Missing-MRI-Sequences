import torch

from brats2023_dataset import BraTS2023Dataset
from Jnetworks.superlightnet import NormalU_Net


DATA_ROOT = (
    r"datasets\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

VAL_SPLIT = r"splits\brats2023\val.txt"


def main():

    print("Loading BraTS 2023 subject...")

    dataset = BraTS2023Dataset(
        data_root=DATA_ROOT,
        split_file=VAL_SPLIT,
        training=False,
    )

    image, label, subject_id = dataset[0]

    # Add batch dimension:
    # [4,128,128,128] -> [1,4,128,128,128]
    image = image.unsqueeze(0)

    print("\nSubject:", subject_id)
    print("Input shape:", image.shape)
    print("Label shape:", label.shape)

    print("\nCreating SuperLightNet...")

    model = NormalU_Net(
        depths_unidirectional="small"
    )

    model.eval()

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", parameters)
    print(
        "Million parameters:",
        parameters / 1e6
    )

    print("\nRunning forward pass on CPU...")
    print("This may take a while.")

    with torch.no_grad():
        output = model(image)

    print("\nSUCCESS!")
    print("Output shape:", output.shape)

    prediction = torch.argmax(
        output,
        dim=1
    )

    print(
        "Prediction shape:",
        prediction.shape
    )

    print(
        "Predicted classes:",
        torch.unique(prediction)
    )


if __name__ == "__main__":
    main()