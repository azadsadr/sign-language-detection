import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import preprocessing

def preprocess(image_path):
    image = Image.open(image_path) # Read a PIL image
    image = ImageOps.grayscale(image)

    # define a transform to 
    transform = transforms.Compose(
        [
            transforms.PILToTensor(), # convert PIL image to  torch tensor
            transforms.Resize(28),
        ]
    )

    #transforms.CenterCrop((28, 28)),
    #transforms.RandomHorizontalFlip(),
    #transforms.ToTensor(),
    #transforms.Normalize([0.5],[0.5])

    img_tensor = transform(image)
    img_tensor = img_tensor.view([1,1,28,28])

    # print the converted Torch tensor
    return img_tensor


def GEN_DATA():
    train = r'data/sign_mnist_train.csv'
    test = r'data/sign_mnist_test.csv'
    train_data, val_data, test_data = preprocessing.make_dataset(
        train_path=train, test_path=test
        )
    labels_map = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 
        6: "G", 7: "H", 8: "I", 9: "K", 10: "L", 11: "M",
        12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S",
        18: "T", 19: "U", 20: "V", 21: "W", 22: "X", 23: "Y",
        }
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # Display image and label.
    train_features, train_labels = next(iter(train_loader))
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {labels_map[label.item()]}")

    return img.view(1,1,28,28), labels_map[label.item()]