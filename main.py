import numpy as np
import torch
from torchvision import datasets, transforms
import os
from networks import Classifier_DropOut, Classifier_noDropOut
from utility import train_model


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

model = Classifier_noDropOut()

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

'''
another optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)
'''

epochs = 200

train_losses_ndp, test_losses_ndp, test_accuracy_ndp = train_model(model, optimizer, criterion, trainloader, testloader, epochs)

model = Classifier_DropOut()

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

train_losses_dp, test_losses_dp, test_accuracy_dp = train_model(model, optimizer, criterion, trainloader, testloader, epochs)

print('end')