import torch.optim as optim
import torch.nn as nn
import torch
from torch.amp import autocast
import sys

def training_step(model, trainloader, epoch):
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
   running_loss = 0.0
   #amp_enabled = amp_dtype in (torch.float16, torch.bfloat16)
   #with autocast(device_type, enabled=amp_enabled, dtype=amp_dtype):
   scaler = torch.cuda.amp.GradScaler()
   for i, data in enumerate(trainloader, 0):
    #print("epoch "+str(i))
    
    
    # Get the inputs; data is a list of [inputs,labels]
    device = torch.device("cuda")
    #data = data.to(device)
    
    inputs, labels = data
    #print(len(inputs))
    labels = labels.to(device)
    inputs = inputs.to(device)
    with torch.cuda.amp.autocast():
    #sys.exit()
    # Zero the parameter gradients
     
     # Forward + backward + optimize
     #inputs = inputs.half()
     outputs = model(inputs)
     #print(inputs.dtype)
     #sys.exit()
     #print(outputs.dtype)
     #sys.exit()
     loss = criterion(outputs, labels)
    #print(loss.dtype)
    #sys.exit()
    
    '''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    '''
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Print statistics
    running_loss += loss.item()
    if i % 2000 == 1999: # Print every 2000 minibatches
       print('[%d, %5d] loss: %.3f' % (epoch + 1,i + 1, running_loss / 2000))
       running_loss = 0.0
   print('Epoch ', epoch, 'finished training')



