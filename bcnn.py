import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
from pyro.nn import PyroSample
from pyro.nn import PyroModule

import pyro.distributions as dist


#prior_dist = distribution.expand(param.shape).to_event(param.dim())

class BCNN(PyroModule):
    def __init__(self):
        super(BCNN, self).__init__()

        distribution = dist.Normal(torch.zeros(1, device='cpu'), torch.ones(1, device='cpu'))

        self.conv1 = PyroModule[nn.Conv2d](in_channels=1, out_channels=28, kernel_size=(3,3), stride=1, padding=1)
        self.conv1.weight = PyroSample(distribution.expand(self.conv1.weight.shape).to_event(self.conv1.weight.dim()))
        self.conv1.bias = PyroSample(distribution.expand(self.conv1.bias.shape).to_event(self.conv1.bias.dim()))

        self.conv2 = PyroModule[nn.Conv2d](in_channels=28, out_channels=28, kernel_size=(3,3), stride=1, padding=1)
        self.conv2.weight = PyroSample(distribution.expand(self.conv2.weight.shape).to_event(self.conv2.weight.dim()))
        self.conv2.bias = PyroSample(distribution.expand(self.conv2.bias.shape).to_event(self.conv2.bias.dim()))

        self.pool1 = PyroModule[nn.MaxPool2d](kernel_size=(2,2), stride=2)

        self.conv3 = PyroModule[nn.Conv2d](in_channels=28, out_channels=56, kernel_size=(3,3), stride=1, padding=1)
        self.conv3.weight = PyroSample(distribution.expand(self.conv3.weight.shape).to_event(self.conv3.weight.dim()))
        self.conv3.bias = PyroSample(distribution.expand(self.conv3.bias.shape).to_event(self.conv3.bias.dim()))

        self.conv4 = PyroModule[nn.Conv2d](in_channels=56, out_channels=56, kernel_size=(3,3), stride=1, padding=1)
        self.conv4.weight = PyroSample(distribution.expand(self.conv4.weight.shape).to_event(self.conv4.weight.dim()))
        self.conv4.bias = PyroSample(distribution.expand(self.conv4.bias.shape).to_event(self.conv4.bias.dim()))

        self.pool2 = PyroModule[nn.MaxPool2d](kernel_size=(2,2), stride=2)

        self.fc1 = PyroModule[nn.Linear](56*7*7, 512)
        self.fc1.weight = PyroSample(distribution.expand(self.fc1.weight.shape).to_event(self.fc1.weight.dim()))
        self.fc1.bias = PyroSample(distribution.expand(self.fc1.bias.shape).to_event(self.fc1.bias.dim()))

        self.fc2 = PyroModule[nn.Linear](512, 128)
        self.fc2.weight = PyroSample(distribution.expand(self.fc2.weight.shape).to_event(self.fc2.weight.dim()))
        self.fc2.bias = PyroSample(distribution.expand(self.fc2.bias.shape).to_event(self.fc2.bias.dim()))

        self.fc3 = PyroModule[nn.Linear](128, 24)
        self.fc3.weight = PyroSample(distribution.expand(self.fc3.weight.shape).to_event(self.fc3.weight.dim()))
        self.fc3.bias = PyroSample(distribution.expand(self.fc3.bias.shape).to_event(self.fc3.bias.dim()))


    def forward(self, x, y):
        output = self.conv1(x)
        output = F.relu(output)
        output = self.conv2(output)
        output = F.relu(output)
        output = self.pool1(output)
        output = self.conv3(output)
        output = F.relu(output)
        output = self.conv4(output)
        output = F.relu(output)
        output = self.pool2(output)
        output = output.view(-1, 56*7*7)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        logits = self.fc3(output)
        
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits
    

def train(train_loader, valid_loader, model, num_epochs, learning_rate, device):

    train_losses = []
    valid_losses = []
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_total_steps = len(train_loader)

    # An epoch is one pass over the entire train set
    for epoch in range(num_epochs): # no. of full passes (loop) over the data

        train_loss = 0.0
        valid_loss = 0.0

        # training the model
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):

            # get data as a list of [images, labels]
            # loader is a batch of featuresets and labels
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
            
            # forward propagation (pass the inputs into neural network)
            outputs = model(images) # (batch_size x num_classes)

            # passing our ouputs (a flattened layer of logits) from the network into a log softmax function and negative log likelihood.
            # All this allows us to get the prediction error (loss) of our network.
            # Note that outputs is our input (the predicted class), and labels is our target (the correct class).
            loss = criterion(outputs, labels)
            #loss = F.nll_loss(output, y)    # calc and grab loss value

            # zero previous parameter gradients. you will do this likely every step
            # It’s a crucial step to zero out the gradients or else all the gradients from multiple passes will accumulated
            optimizer.zero_grad()

            # performing back-propagation by computing the gradient based on the loss.
            loss.backward()

            # gradient descent or adam step (optimize weights)
            # After computing gradient using backward(), 
            # we can call the optimizer step function, 
            # which iterates over all the parameters and update their values.
            optimizer.step()

            # item() extracts the loss’s value as a Python float. 
            # We then add it to our train_loss (which is zero at the start of every iteration)
            train_loss += loss.item() * images.size(0)

            if (batch_idx+1) % 200 == 0:
                print(f'epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{n_total_steps}], loss = {loss.item():.4f}')
        
        #print("==============================================================")
        # validate the model
        model.eval()
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print-training/validation-statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))


    return model, train_losses, valid_losses



def get_accuracy(loader, model, device):
    
    num_correct = 0
    num_total = 0

    model.eval()
    # Test the model. we don't need to compute gradients (for memory efficiency)
    # torch.no_grad is used when we don’t require PyTorch to run its autograd engine, 
    # in other words, calculate the gradients of our input. Since we’re only calculating the accuracy of our network. 
    # This will help reduce memory usage and speed up computation.
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1) # max returns (value, index)
            
            # this is a boolean expression. 
            # We can sum the amount of times we get the right prediction, 
            # and then grab the numeric value using item()
            num_correct += (predictions == labels).sum().item()

            num_total += labels.shape[0]
        
        print(f'Correct: [{num_correct} / {num_total}] with accuracy {float(num_correct)/float(num_total)*100:.2f} %')
