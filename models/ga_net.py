import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        bn1 = nn.BatchNorm2d(num_features=mid_channels)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1)
        bn2 = nn.BatchNorm2d(num_features=out_channels)
        relu2 = nn.ReLU(inplace=True)
        super().__init__(conv1, bn1, relu1, conv2, bn2, relu2)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conn_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_conn_channels, out_channels)

    def forward(self, x, skip_features):
        x = self.up(x)
        diff_y = skip_features.size()[2] - x.size()[2]
        diff_x = skip_features.size()[3] - x.size()[3]
        if diff_y != 0 or diff_x != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip_features, x], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, width_factor=64, blocks=5):
        super().__init__()
        channels = [width_factor << i for i in range(blocks)]

        block_0 = DoubleConv(in_channels=in_channels, out_channels=channels[0])
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels=channels[i - 1], out_channels=channels[i]))
            for i in range(1, blocks)])

        self.blocks.insert(0, block_0)
        self.out_channels = channels

    def forward(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class GADecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        in_channels = encoder_channels[::-1]
        self.area_blocks = nn.ModuleList([
            Up(in_channels=in_channels[i], out_channels=in_channels[i + 1],
               skip_conn_channels=in_channels[i + 1])
            for i in range(0, len(in_channels) - 1)])

        self.tissue_blocks = nn.ModuleList([
            Up(in_channels=in_channels[i], out_channels=in_channels[i + 1],
               skip_conn_channels=in_channels[i + 1])
            for i in range(0, len(in_channels) - 1)])

        self.ag_start_block = 1
        self.gate_blocks = nn.ModuleList(
            [AttentionGate(in_channels[i + 1], in_channels[i + 1])
             for i in range(self.ag_start_block, len(in_channels) - 1)])

    def forward(self, x):
        skip_connections = x[-2::-1]
        x_tissue = x[-1]
        x_area = x[-1]
        for i, skip_connection in enumerate(skip_connections):
            x_tissue = self.tissue_blocks[i](x_tissue, skip_connection)
            x_area = self.area_blocks[i](x_area, skip_connection)
            if i >= self.ag_start_block:
                x_tissue = self.gate_blocks[i - self.ag_start_block](x_tissue, x_area)
        return x_tissue, x_area


class AttentionGate(nn.Module):
    def __init__(self, x_channels, g_channels):
        super().__init__()
        attn_channels = (g_channels + x_channels) // 2
        self.conv_g = nn.Sequential(nn.Conv2d(g_channels, attn_channels, 1),
                                    nn.BatchNorm2d(attn_channels))

        self.conv_x = nn.Sequential(nn.Conv2d(x_channels, attn_channels, 1),
                                    nn.BatchNorm2d(attn_channels))

        self.psi = nn.Sequential(nn.Conv2d(attn_channels, 1, 1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g = self.conv_g(g)
        x_g = self.conv_x(x)
        g = self.relu(g + x_g)
        g = self.psi(g)

        return g * x


class GANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNetEncoder(in_channels=1)
        self.decoder = GADecoder(self.encoder.out_channels)

        self.area_head_1 = DoubleConv(64, 32)
        self.area_head_2 = DoubleConv(32, 2)

        self.tissue_head_1 = DoubleConv(64, 32)
        self.tissue_head_2 = DoubleConv(32, 2)

        self.attn_gate1 = AttentionGate(32, 32)
        self.attn_gate2 = AttentionGate(2, 2)

    def forward(self, x):
        x = self.encoder(x)
        tissue, area = self.decoder(x)
        area = self.area_head_1(area)
        tissue = self.tissue_head_1(tissue)
        tissue = self.attn_gate1(tissue, area)

        area = self.area_head_2(area)
        tissue = self.tissue_head_2(tissue)
        tissue = self.attn_gate2(tissue, area)
        return tissue, area
