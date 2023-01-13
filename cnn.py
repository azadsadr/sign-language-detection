import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        '''
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ... )
            in_channels (int)                       : Number of channels in the input image
            out_channels (int)                      : Number of channels produced by the convolution
            kernel_size (int or tuple)              : Size of the convolving kernel
            stride (int or tuple, optional)         : Stride of the convolution. Default: 1
            padding (int, tuple or str, optional)   : Padding added to all four sides of the input. Default: 0

        torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, ... )
            kernel_size : the size of the window to take a max over
            stride      : the stride of the window. Default value is kernel_size
            padding     : implicit zero padding to be added on both sides

        torch.nn.Dropout2d(p=0.5, inplace=False)
            p (float, optional) : probability of an element to be zero-ed.

        torch.nn.Linear(in_features, out_features, bias=True, ... )
            in_features     : size of each input sample
            out_features    : size of each output sample
            bias            : If set to False, the layer will not learn an additive bias. Default: True
        '''

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=28, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=28, out_channels=28, kernel_size=(3,3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=(3,3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=(3,3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        #self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(56*7*7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 24)

        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
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
        output = self.fc3(output)
        #output = self.dropout1(output)

        # When we don’t know how many rows or columns you want, 
        # PyTorch can automatically set a value for you when you pass in -1. 
        # In our case, we know our columns will be 30 * 3 * 3, but we don’t know how many rows we want.
        
        #output = self.softmax(output)
        return output
    

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
            train_loss += loss.item() * images.shape[0]

            '''
            if (batch_idx+1) % 200 == 0:
                print(f'epoch [{epoch+1}/{num_epochs}], batch [{batch_idx+1}/{n_total_steps}], loss = {loss.item():.4f}')
            '''
        
        #print("==============================================================")
        # validate the model
        model.eval()
        for batch_idx, (images, labels) in enumerate(valid_loader):
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.shape[0]

        # calculate average losses
        train_loss_avg = train_loss / len(train_loader.sampler)
        valid_loss_avg = valid_loss / len(valid_loader.sampler)
        train_losses.append(train_loss_avg)
        valid_losses.append(valid_loss_avg)

        # print-training/validation-statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss_avg, valid_loss_avg))

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
