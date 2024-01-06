import os
import torch
from tests import _PATH_DATA
import Test1.data.make_dataset as DataMaker
import logging

LOGGER = logging.getLogger(__name__)

N_train = 50000
N_test = 5000
# Create the dataset
DataMaker.make_dataset()

train_image_path = os.path.join(_PATH_DATA, "processed", "train_imgs_v2.pt")
train_label_path = os.path.join(_PATH_DATA, "processed", "train_labels_v2.pt")
test_image_path  = os.path.join(_PATH_DATA, "processed", "test_imgs_v2.pt")
test_labels_path = os.path.join(_PATH_DATA, "processed", "test_labels_v2.pt")



def test_data():
    train_images = torch.load(train_image_path)
    train_labels = torch.load(train_label_path)
    val_images = torch.load(test_image_path)
    val_labels = torch.load(test_labels_path)

    assert len(train_images) == N_train
    assert len(val_images) == N_test

    assert len(train_images) == len(train_labels)
    assert len(val_images) == len(val_labels)

    assert 10 == 10, "Random test"

    for img in train_images:
        assert img.shape[0] == 28*28
        break
