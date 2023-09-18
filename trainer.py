import torch.optim as optim
import torch.nn as nn

def training_step(model, trainloader, epoch):
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
   running_loss = 0.0
   for i, data in enumerate(trainloader, 0):
    # Get the inputs; data is a list of [inputs,labels]
    inputs, labels = data
    # Zero the parameter gradients
    optimizer.zero_grad()
    # Forward + backward + optimize
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    # Print statistics
    running_loss += loss.item()
    if i % 2000 == 1999: # Print every 2000 minibatches
       print('[%d, %5d] loss: %.3f' % (epoch + 1,i + 1, running_loss / 2000))
       running_loss = 0.0
   print('Epoch ', epoch, 'finished training')



