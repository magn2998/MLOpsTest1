import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from Test1.models.model import MyCnnNetwork, MyNeuralNet
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


EPOCHS = int(sys.argv[1])

model = MyCnnNetwork()
print(model)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)

# Create Data Loaders and Load Data Sets
train_images = torch.load("data/processed/train_imgs_v2.pt")
train_labels = torch.load("data/processed/train_labels_v2.pt")
val_images = torch.load("data/processed/test_imgs_v2.pt")
val_labels = torch.load("data/processed/test_labels_v2.pt")

train_dataset = TensorDataset(train_images, train_labels)  # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # create your dataloader

val_dataset = TensorDataset(val_images, val_labels)  # create your datset
val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)  # create your dataloader

# Define Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="./models",
    monitor="val_accuracy",
    mode="max",
    filename="LightningTrainedModel.pt",
    save_on_train_epoch_end=True,
)


trainer = Trainer(callbacks=[checkpoint_callback], max_epochs=10)
trainer.fit(model, train_dataloader, val_dataloader)

print("Training Done")
