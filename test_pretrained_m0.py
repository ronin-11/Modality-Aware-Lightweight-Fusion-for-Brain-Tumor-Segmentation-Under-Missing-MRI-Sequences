import torch

from brats2023_dataset import BraTS2023Dataset
from Jnetworks.superlightnet import NormalU_Net


DATA_ROOT = (
    r"datasets\raw\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)

VAL_SPLIT = r"splits\brats2023\val.txt"
CHECKPOINT = "model.pth"


def main():

    # ---------------------------------------
    # Dataset
    # ---------------------------------------
    print("Loading BraTS 2023 subject...")

    dataset = BraTS2023Dataset(
        data_root=DATA_ROOT,
        split_file=VAL_SPLIT,
        training=False,
    )

    image, label, subject_id = dataset[0]

    image = image.unsqueeze(0)

    print("Subject:", subject_id)
    print("Input:", image.shape)

    # ---------------------------------------
    # Model
    # ---------------------------------------
    print("\nCreating SuperLightNet...")

    model = NormalU_Net(
        depths_unidirectional="small"
    )

    # ---------------------------------------
    # Official checkpoint
    # ---------------------------------------
    print("Loading official model.pth...")

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu"
    )

    # Official checkpoint was saved using
    # nn.DataParallel, hence the "module." prefix.
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    result = model.load_state_dict(
        state_dict,
        strict=True
    )

    print("Checkpoint loaded successfully.")
    print(result)

    model.eval()

    # ---------------------------------------
    # Forward pass
    # ---------------------------------------
    print("\nRunning pretrained forward pass...")

    with torch.no_grad():
        output = model(image)

    prediction = torch.argmax(
        output,
        dim=1
    )

    print("\nSUCCESS!")
    print("Output:", output.shape)
    print("Prediction:", prediction.shape)
    print(
        "Predicted classes:",
        torch.unique(prediction)
    )


if __name__ == "__main__":
    main()