import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv4(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, output_size=5):
        super(Conv4, self).__init__()
        linear_features = 1600 if z_dim == 64 else 800 if z_dim == 32 else None
        self.encoder = nn.Sequential(self.conv_block3(x_dim, hid_dim), self.conv_block3(hid_dim, hid_dim),
                                 self.conv_block3(hid_dim, hid_dim), self.conv_block3(hid_dim, z_dim),
                                 Flatten())
        self.classifier = nn.Linear(in_features=linear_features, out_features=output_size, bias=True)

    def conv_block3(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def conv_block1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

class Conv4Encoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(Conv4Encoder, self).__init__()
        self.encoder = nn.Sequential(self.conv_block3(x_dim, hid_dim), self.conv_block3(hid_dim, hid_dim),
                                 self.conv_block3(hid_dim, hid_dim), self.conv_block3(hid_dim, z_dim), 
                                 Flatten())

    def conv_block3(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.encoder(x)
