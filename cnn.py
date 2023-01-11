import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3,3))
        self.dropout1 = nn.Dropout2d()

        self.fc1 = nn.Linear(30*3*3, 270)
        self.fc2 = nn.Linear(270, 24)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = F.relu(output)
        output = self.pool2(output)

        output = self.conv3(output)
        output = F.relu(output)
        output = self.dropout1(output)

        output = output.view(-1, 30*3*3)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        #output = self.softmax(output)
        return output


def train(loader, model, num_epochs, learning_rate, device):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(loader)
    model.train()
    for epoch in range(num_epochs): # no. of full passes (loop) over the data
        #running_loss = 0
        #print(f'epoch: {epoch+1}')

        for batch_idx, (images, labels) in enumerate(loader):

            # get data as a list of [images, labels]
            # train_loader is a batch of featuresets and labels
            # batch_idx : index of the batch
            # images    : one batch of features
            # labels    : one batch of targets
            
            # get data to cuda if possible
            images = images.to(device=device)
            labels = labels.to(device=device)
            
            # images are in correct shape
            # no need to flatten MNIST images like normal neural network
            # we did it inside CNN class
            #images = print("images.shape:", images.shape)
            
            # forward propagation
            outputs = model(images) # (batch_size x num_classes)
            
            #output = net(X.view(-1, 28*28)) # pass in reshaped batch

            loss = criterion(outputs, labels)
            #loss = F.nll_loss(output, y)    # calc and grab loss value

            # zero previous gradients. you will do this likely every step
            optimizer.zero_grad()
            # back-propagation
            loss.backward()
            # gradient descent or adam step (optimize weights)
            optimizer.step()

            #running_loss += loss.item()

            if (batch_idx+1) % 200 == 0:
                print(f'epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{n_total_steps}], loss = {loss.item():.4f}')

        #print("loss =", loss.item()) # print loss. we hope loss (a measure of wrong-ness) declines!
        print("==============================================================")

    return model


def get_accuracy(loader, model, device):
    
    num_correct = 0
    num_total = 0

    model.eval()
    # Test the model. we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            # images are in correct shape
            # no need to flatten MNIST images like normal neural network
            # we did it inside CNN class
            
            outputs = model(images)
            
            _, predictions = torch.max(outputs, 1) # max returns (value, index)
            #_, predictions = outputs.max(1)
            num_correct += (predictions == labels).sum().item()
            num_total += labels.shape[0]
        
        print(f'Correct: [{num_correct} / {num_total}] with accuracy {float(num_correct)/float(num_total)*100:.2f} %')