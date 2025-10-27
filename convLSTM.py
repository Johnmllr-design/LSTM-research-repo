import torch
import torch.nn as nn
from convLSTMcell import ConvLSTMCell



class ConvLSTM(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=32):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels)               # cell, which [produces 32 outputs and updates gates
        self.output_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)         # output conv to go from 32 channels to 1 channel

    def forward(self, x, hidden=None):
        # x: [B, C, H, W] — single frame at t
        B, C, H, W = x.shape

        if hidden is None:
            h = torch.zeros(B, 32, H, W, device=x.device)
            c = torch.zeros(B, 32, H, W, device=x.device)
        else:
            h, c = hidden

        h, c = self.cell(x, h, c)
        output = self.output_conv(h)  # logits
        output = torch.sigmoid(output) # [B, 1, H, W]
        return output, (h, c)