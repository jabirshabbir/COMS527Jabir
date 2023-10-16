import torch
import torch.nn as nn
import torch.optim as optim
import time
from DummyModel import DummyModel 
model = DummyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(30, 10))
labels = []
inputs = []
for i in range(0,500):
   inputs.append(torch.randn(30,10))
   label = torch.randn(30,10).to('cuda:1')
   labels.append(label)

start_time = time.time()
for epoch in range(0,10):
    for i in range(0,len(inputs)):
       outputs = model(inputs[i])
       loss_fn(outputs, labels[i]).backward()
       optimizer.step()

end_time = time.time()
print(end_time-start_time)
print('done')
