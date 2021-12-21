# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)

class net1(torch.nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        # self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(2)])
        abc = nn.ModuleList([nn.Linear(10, 10) for i in range(2)])
        self.linears = abc.extend(nn.ModuleList(nn.Linear(10, 10)))

    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x

net = net1()
print(net)
for param in net.parameters():
    print(type(param.data), param.size())
    print(f"{'-'*20}")

# %%
class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(5, 7),
                                     nn.Linear(7, 10),
                                     nn.Linear(3, 5)])
    def forward(self, x):
        x = self.linears[2](x)
        x = self.linears[0](x)
        x = self.linears[1](x)
        return x

net = net2()
print(net)
input = torch.rand(2, 3)
print(input)
print(net(input).shape)
print(net(input))

# %%
class net3(nn.Module):
    def __init__(self):
        super(net3, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(1,4,2),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 8, 2),
                                   nn.ReLU())
    def forward(self,x):
        x = self.block(x)
        return x

net = net3()
print(net)
# %%
class net4(nn.Module):
    def __init__(self):
        super(net4, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for i in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

net = net4()
print(net)
# %%
class net5(nn.Module):
    def __init__(self):
        super(net5, self).__init__()
        self.list_layers = [nn.Linear(10, 10) for i in range(2)]
        self.layers = nn.Sequential(*self.list_layers)
    def forward(self, x):
        self.x = self.layers(x)
        return x

net = net5()
print(net)
# %%
class net6(nn.Module):
    def __init__(self):
        super(net6, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 5),
                                     nn.Linear(5,8),
                                     nn.Linear(8,10)])
        self.trace = []
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            self.trace.append(x)
        return x
    
net = net6()
print(net)

for param in net.parameters():
    print(type(param.data), param.size())

input = torch.randn(2, 3)  ## batch size: 2
output = net(input)
for each_layer in net.trace:
    print(each_layer.shape)

# %%
class net7(nn.Module):
    def __init__(self, in_features, nb_classes, nb_hidden_layer, 
        hidden_size, act=nn.ReLU):
        super(net7, self).__init__()
        self.act = act()
        
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])
        self.out = nn.Linear(hidden_size, nb_classes)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        for l in self.fcs:
            x = F.relu(l(x))
        x = self.out(x)
        return x
            
net = net7(2, 3, 4, 5, nn.ReLU)
print(net)
# %%
