import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from Test1.models.model import MyCnnNetwork

import sys

EPOCHS = int(sys.argv[1])

model = MyCnnNetwork()
print(model)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)


# Create Data Loaders and Load Data Sets
train_images = torch.load("data/processed/train_imgs.pt")
train_labels = torch.load("data/processed/train_labels.pt")
test_images = torch.load("data/processed/test_imgs.pt")
test_labels = torch.load("data/processed/test_labels.pt")

train_dataset = TensorDataset(train_images, train_labels)  # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # create your dataloader

test_dataset = TensorDataset(test_images, test_labels)  # create your datset
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)  # create your dataloader

# Load Test Set for validation
dataiter = iter(test_dataloader)
test_images, test_labels = next(dataiter)

train_loss = []
test_acc = []
for epoch in range(EPOCHS):
    print("Running Epoch no. " + str(epoch))
    for images, labels in train_dataloader:
        # Flatten Corrupted MNIST images
        # print(images.shape)

        optimizer.zero_grad()

        # TODO: Training pass
        output, _ = model.forward(images)
        # print(output)
        # print(output.shape)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()

        train_loss.append(loss)

    else:
        model.eval()
        # Use Data
        # print(test_images.shape)
        # print(test_labels.shape)

        pred, _ = model.forward(test_images)

        # print(pred)
        ps = torch.exp(pred)

        _, top_class = ps.topk(1, dim=1)

        correct_guesses = top_class == test_labels.view(*top_class.shape)
        accuracy = torch.mean(correct_guesses.type(torch.FloatTensor))
        print(f"Accuracy for epoch {epoch}: {accuracy.item()*100}%")

        test_acc.append(accuracy.item() * 100)

        # Set model back to train mode
        model.train()
        if epoch == EPOCHS - 1:
            # Create a subplot with 5 rows and 2 columns
            fig, axs = plt.subplots(5, 2, figsize=(12, 30))

            for i in range(5):
                test_image = test_images[i].view(28, 28)
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
            plt.savefig("reports/figures/Result2.png")
            torch.save(model.state_dict(), "models/TrainedModel.pt")
