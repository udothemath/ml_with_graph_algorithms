# %%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# %%

## Q
## 1. In __init__(), what does it mean in Conv2d() and Linear() argument?

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ## Max pooling over a (2, 2) window
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # ## 2 is ame as (2, 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # x = x.view(-1, self.num_flat_features(x))
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        x =  self.check_features(x)
        print("hello from forward")
        return x

    def check_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        # num_features = 1
        # for s in size:       # Get the products
        #     num_features *= s
        # return num_features
        return size


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features

net = Net().to(device)
print(net)
# Net(
#  (conv1): Conv2d (1, 6, kernel_size=(5, 5), stride=(1, 1))
#  (conv2): Conv2d (6, 16, kernel_size=(5, 5), stride=(1, 1))
#  (fc1): Linear(in_features=400, out_features=120)
#  (fc2): Linear(in_features=120, out_features=84)
#  (fc3): Linear(in_features=84, out_features=10)
#)

# %%
params = list(net.parameters())
print(len(params))       # 10: 10 sets of trainable parameters

print(params[0].size())  # torch.Size([6, 1, 5, 5])

# %%

x = Variable(torch.randn(1, 1, 32, 32))
print(x.size())
size = x.size()[3:]  # all dimensions except the batch dimension
print(size)

# input = Variable(torch.randn(1, 1, 32, 32))
# out = net(input)   # out's size: 1x10.
#print(out)
# Variable containing:
# 0.1268  0.0207  0.0857  0.1454 -0.0370  0.0030  0.0150 -0.0542  0.0512 -0.0550
# [torch.FloatTensor of size 1x10]

# %%
# a = torch.randn(1, 1, 32, 32)
# print(a)

a = torch.randn(2, 3)
print(a)

a = torch.randn(1, 2, 3)
print(a)


# %%
