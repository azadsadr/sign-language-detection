from PIL import Image, ImageOps
import torchvision.transforms as transforms

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
    