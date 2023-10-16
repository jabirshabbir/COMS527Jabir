import torch
import torch.nn as nn
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net0 = nn.Linear(10, 30).to('cuda:0')
        self.relu = nn.ReLU()
        self.net1 = nn.Linear(30, 10).to('cuda:0')
        self.net2 = nn.Linear(10,20).to('cuda:0')
        self.net3 = nn.Linear(20,7).to('cuda:1')
        self.net4 = nn.Linear(7,15).to('cuda:1')
        self.net5 = nn.Linear(15,10).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net0(x.to('cuda:0')))
        x = self.relu(self.net1(x))
        x = self.relu(self.net2(x))
        x = self.relu(self.net3(x.to('cuda:1')))
        x = self.relu(self.net4(x))
        x = self.relu(self.net5(x))
        return x
