import torch
import torch.nn as nn
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.net0 = nn.Linear(20, 10).to('cuda:0')
        self.relu = nn.ReLU()
        self.net1 = nn.Linear(10, 10).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net0(x.to('cuda:0')))
        return self.net1(x.to('cuda:1'))
