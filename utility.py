import torch


def train_model(model, optimizer, criterion, trainloader, testloader, epochs):
    train_losses, test_losses, test_accuracy = [], [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images.view(images.shape[0],-1))
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            test_loss = 0
            accuracy = 0
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                # set model to evaluation mode
                model.eval()
                for images, labels in testloader:
                    log_ps = model(images.view(images.shape[0],-1))
                    test_loss += criterion(log_ps, labels)
                    
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            # set model back to train mode
            model.train()        
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            test_accuracy.append(accuracy/len(testloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    else:
        return train_losses, test_losses, test_accuracy

        
