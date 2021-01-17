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




'''
another possible implementation would be:
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

model = Classifier()
'''
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

'''
another optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)
'''

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

#accuracy calculation on the test set
with torch.no_grad():
    logps = model.forward(images.view(images.shape[0],-1))
ps = torch.exp(logps)
top_ps, top_classes= ps.topk(1)

accuracy=torch.mean((top_classes==labels.view(*top_classes.shape)).type(torch.FloatTensor))

print(accuracy)

