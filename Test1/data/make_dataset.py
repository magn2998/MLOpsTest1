import torch

if __name__ == "__main__":
    # exchange with the corrupted mnist dataset
    path_train_str = 'data/raw/corruptmnist_v2/'
    path_test_str  = 'data/raw/corruptmnist/'

    test_images = torch.load(path_test_str + "test_images.pt")
    test_labels = torch.load(path_test_str + "test_target.pt")

    train_images = torch.load(path_test_str + "train_images_0.pt")
    train_labels = torch.load(path_test_str + "train_target_0.pt")

    train_images = torch.cat((train_images, torch.load(path_test_str + "train_images_1.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_test_str + "train_target_1.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_test_str + "train_images_2.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_test_str + "train_target_2.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_test_str + "train_images_3.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_test_str + "train_target_3.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_test_str + "train_images_4.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_test_str + "train_target_4.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_test_str + "train_images_5.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_test_str + "train_target_5.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_train_str + "train_images_6.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_train_str + "train_target_6.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_train_str + "train_images_7.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_train_str + "train_target_7.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_train_str + "train_images_8.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_train_str + "train_target_8.pt")), 0)

    train_images = torch.cat((train_images, torch.load(path_train_str + "train_images_9.pt")), 0)
    train_labels = torch.cat((train_labels, torch.load(path_train_str + "train_target_9.pt")), 0)

    # Flatten Images
    print(test_images.shape)
    test_images = test_images.view(test_images.shape[0], -1)
    train_images = train_images.view(train_images.shape[0], -1)
    # Normalize

    # Save Data
    torch.save(train_images, "data/processed/train_imgs_v2.pt")
    torch.save(train_labels, "data/processed/train_labels_v2.pt")
    torch.save(test_images, "data/processed/test_imgs_v2.pt")
    torch.save(test_labels, "data/processed/test_labels_v2.pt")
