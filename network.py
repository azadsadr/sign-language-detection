import torch.nn as nn

def network(in_channels, output_size, device):
    
    # in_channels=1
    # output_size=24

    activ = nn.LeakyReLU

    net = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5,),
        activ(),
        nn.MaxPool2d(2,2),
        nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3,),
        activ(),
        nn.MaxPool2d(2,2),
        nn.Flatten(),
        nn.Linear(512*5*5, 256),
        activ(),
        nn.Linear(256, output_size)
        )
    net = net.to(device)

    return net