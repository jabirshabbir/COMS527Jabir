from data_handler import load_cifar10
from model_architecture import Net
from trainer import training_step
from evaluate import evaluate
import time
import torch
import torch.nn as nn
#from deepspeed.pipe import PipelineModule
import sys

if __name__ == '__main__':
    n_epochs = 10
    #print(torch.cuda.is_available())
    #sys.exit()
    model = Net()
    device = torch.device("cuda")
    model = model.to(device)
    model = nn.DataParallel(model)
    device = next(model.parameters()).device
    print(device)
    #sys.exit()
    trainloader,testloader,classes = load_cifar10()

    for epoch in range(n_epochs):
      start_time = time.time();
      print("current epoch is "+str(epoch))
      training_step(model, trainloader, epoch)
      evaluate(model, testloader,trainloader)
      end_time = time.time();
      print(end_time-start_time)
      sys.exit()
    print("-"*10,"Training finshed","-"*10)