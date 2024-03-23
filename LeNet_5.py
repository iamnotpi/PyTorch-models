from torch import nn

class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 6, 5, 1, padding=2) # The original LeNet-5 expects images of size 32x32 as inputs, but in the MNIST dataset the images are of size 28x28
        self.avgpool_1 = nn.AvgPool2d(2, 2)
        self.conv_2 = nn.Conv2d(6, 16, 5, 1)
        self.avgpool_2 = nn.AvgPool2d(2, 2)
        self.conv_3 = nn.Conv2d(16, 120, 5, 1)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(120, 84)
        self.linear_2 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.relu(self.conv_1(x))
        z = self.avgpool_1(z)
        z = self.relu(self.conv_2(z))
        z = self.avgpool_2(z)
        z = self.relu(self.conv_3(z))
        z = self.flatten(z)
        return self.linear_2(self.relu(self.linear_1(z)))
