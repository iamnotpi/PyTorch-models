from torch import nn

class Conv2dWithActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=nn.ReLU(), **kwargs):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class InceptionModule(nn.Module):
    """ Inception module as described in the paper "Going Deeper with Convolutions" by Szegedy et al.
    The parameters' names are referenced from the paper.

    Args:
        in_channels: Number of input channels
        out_channels_1: Number of output channels for the 1x1 conv layer
        reduce_3x3: Number of output channels for the 1x1 conv layer before the 3x3 conv layer
        out_channels_3: Number of output channels for the 3x3 conv layer
        reduce_5x5: Number of output channels for the 1x1 conv layer before the 5x5 conv layer
        out_channels_5: Number of output channels for the 5x5 conv layer
        pool_proj: Number of output channels for the 1x1 conv layer before the pooling layer
    """
    def __init__(self, in_channels, out_channels_1, reduce_3x3, out_channels_3, reduce_5x5, out_channels_5, pool_proj):
        super().__init__()
        self.conv_1 = Conv2dWithActivation(in_channels, out_channels_1, 1)
        self.reduce_3x3 = Conv2dWithActivation(in_channels, reduce_3x3, 1)
        self.reduce_5x5 = Conv2dWithActivation(in_channels, reduce_5x5, 1)
        self.conv_3 = Conv2dWithActivation(reduce_3x3, out_channels_3, 3, padding=1)
        self.conv_5 = Conv2dWithActivation(reduce_5x5, out_channels_5, 5, padding=2)
        self.pool_proj = Conv2dWithActivation(in_channels, pool_proj, 1)
        self.max_pool = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x):
        return torch.cat((
                self.conv_1(x),
                self.conv_3(self.reduce_3x3(x)),
                self.conv_5(self.reduce_5x5(x)),
                self.pool_proj(self.max_pool(x))
            ), dim=1 # Tensors of shape (batch_size, num_filters, height, width) so dim=1
        )

class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_layer = nn.Sequential(
            Conv2dWithActivation(3, 64, 7, stride=2),
            nn.MaxPool2d(3, 2),
            Conv2dWithActivation(64, 192, 3),
            nn.MaxPool2d(3, 2),
            InceptionModule(192, 64, 96, 128, 16, 32, 32),
            InceptionModule(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2),
            InceptionModule(480, 192, 96, 208, 16, 48, 64),
            InceptionModule(512, 160, 112, 224, 24, 64, 64),
            InceptionModule(512, 128, 128, 256, 24, 64, 64),
            InceptionModule(512, 112, 144, 288, 32, 64, 64),
            InceptionModule(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2),
            InceptionModule(832, 256, 160, 320, 32, 128, 128),
            InceptionModule(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7),
            nn.Dropout(0.4),
            nn.Linear(1024, 10),
        )
    
    def forward(self, x):
        return self.main_layer(x)
