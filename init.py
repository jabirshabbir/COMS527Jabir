from data_handler import load_cifar10
from model_architecture import Net
from trainer import training_step
from evaluate import evaluate
import torch
import sys

if __name__ == '__main__':
    n_epochs = 10
    #print(torch.cuda.is_available())
    #sys.exit()
    model = Net()
    trainloader,testloader,classes = load_cifar10()

    for epoch in range(n_epochs):
      training_step(model, trainloader, epoch)
      evaluate(model, testloader,trainloader)
    print("-"*10,"Training finshed","-"*10)