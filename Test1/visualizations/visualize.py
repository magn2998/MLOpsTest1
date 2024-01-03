import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
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

data_detached = cnnOut.detach()
data_np = data_detached.numpy()
data_reshaped = data_np.reshape(5000, -1)  # The shape becomes [5000, 784]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
data_2d = tsne.fit_transform(data_reshaped)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1])
plt.title("t-SNE Visualization of the Data")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.savefig("reports/figures/PredictionTSNE.png")

print("Done Visualizing")
