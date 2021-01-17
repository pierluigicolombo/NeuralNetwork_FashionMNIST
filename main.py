import numpy as np
import torch
from torchvision import datasets, transforms
import os


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
path = os.path.realpath(__file__).split('main.py')[0]

# Download and load the training data
trainset = datasets.FashionMNIST(path + 'input/trainset/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST(path + 'input/testset/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


model = torch.nn.Sequential(torch.nn.Linear(784, 128),
                      torch.nn.ReLU(),
                      torch.nn.Linear(128, 64),
                      torch.nn.ReLU(),
                      torch.nn.Linear(64, 10),
                      torch.nn.LogSoftmax(dim=1))


criterion = torch.nn.NLLLoss()

'''
another possible implementation would be:
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))
criterion = nn.CrossEntropyLoss()
infact nn.CrossEntropy combines nn.LogSoftmax() and nn.NLLLoss() in one single class.
The input is expected to contain scores for each class.
'''

optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # TODO: Training pass
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")



# Test out your network!

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1].view(1,-1)

# TODO: Calculate the class probabilities (softmax) for img
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model.forward(img) # prediction fase

ps = torch.exp(logps) #with the exponential ps contains the probability for each class
print(ps)
#----------

