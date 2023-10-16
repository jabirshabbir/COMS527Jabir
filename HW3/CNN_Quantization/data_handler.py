import torch
import torchvision
import torchvision.transforms as transforms
# Define transformations for the images

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(),\
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))])
    
    # Download and load training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=10000,shuffle=True,num_workers=2)
    # Download and load test dataset
    #testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,shuffle=True,num_workers=2)
    # Download and load test dataset
    testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset,batch_size=32,shuffle=False,num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader,testloader,classes






