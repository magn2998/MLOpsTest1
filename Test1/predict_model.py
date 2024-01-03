import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from Test1.models.model import MyCnnNetwork
import matplotlib.pyplot as plt
import sys


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    dataiter = iter(test_dataloader)
    test_images, _ = next(dataiter)
    pred, cnnOut = model.forward(test_images)
    return pred, cnnOut, test_images


# Setup Model Name
model_name = sys.argv[1]
model_path = "models/" + str(model_name)
print("Loading Model " + model_path)

# Load Model
model = MyCnnNetwork()
model.load_state_dict(torch.load(model_path))
model.eval()

# Setup DataLoaders
test_images = torch.load("data/processed/test_imgs.pt")
test_labels = torch.load("data/processed/test_labels.pt")

test_dataset = TensorDataset(test_images, test_labels)  # create your datset
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)  # create your dataloader

pred, cnnOut, images = predict(model, test_dataloader)
ps = torch.exp(pred)

print(pred.shape)
print(cnnOut.shape)

# Create a subplot with 5 rows and 2 columns
fig, axs = plt.subplots(5, 2, figsize=(12, 30))

for i in range(5):
    test_image = images[i].view(28, 28)
    ps_numpy = ps[i].data.numpy().squeeze()

    # Display the image
    axs[i, 0].imshow(test_image, cmap="gray")
    axs[i, 0].set_title(f"Test Image {i+1}")

    # Bar diagram for the predictions
    axs[i, 1].bar(np.arange(len(ps_numpy)), ps_numpy)
    axs[i, 1].set_title(f"Predictions for Image {i+1}")
    axs[i, 1].set_xlabel("Class")
    axs[i, 1].set_ylabel("Probability")

# Adjust the layout
plt.tight_layout()

# Save the figure and model
plt.savefig("reports/figures/Prediction.png")
