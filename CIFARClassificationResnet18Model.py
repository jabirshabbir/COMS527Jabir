import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import sys
from evaluateCIFARModel import evaluate
import torch.nn as nn

# Define data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

import datetime


# Load CIFAR-10 data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# Define ResNet-18 model
net = torchvision.models.resnet18(pretrained=True, num_classes=1000)
#print(len(net.parameters()))
#sys.exit()
#print(net.layer1.trainable)
#newmodel = torch.nn.Sequential(*(list(net.children())[:]))
#print(newmodel)

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 10)
#sys.exit()

no_of_modules = 0
for  module in net.parameters():
   no_of_modules = no_of_modules +1

#no_of_modules = len(net.named_modules())

index =0
modules_new = []
#no_of_modules = index
index =0
for module in net.parameters():
  module.requires_grad = False
  index = index+1

#sys.exit()
#net_new = nn.Sequential()

'''
for  module in modules_new:
    #print(f"module_name : {module_name} , value : {module}")
    net_new.add(module)
    print(module)
sys.exit()
'''
# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
PATH = "CifarResnetModel.pt"
checkpoint = torch.load(PATH)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Train the model
for epoch in range(200):
    net.train()
    running_loss = 0.0
    print(epoch)
    print(datetime.datetime.now())
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(datetime.datetime.now())
    print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
    if epoch %10 ==0:
      evaluate(net, testloader,trainloader)
    torch.save({
          'epoch': epoch,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': running_loss,
          }, PATH)
# Test the model
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test set: %d %%' % (100 * correct / total))