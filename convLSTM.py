import torch
import torch.nn as nn
from convLSTMcell import ConvLSTMCell


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM Network for Spatial-Temporal Processing.
    
    This module combines a ConvLSTM cell with an output projection layer to process
    sequential spatial data. It's designed for spacial-temporal bound microbubble
    detection.
    
    The network processes input through a ConvLSTM cell and projects the output to
    a single-channel prediction map using sigmoid activation for per-pixel predictions.
    
    Args:
        input_channels (int, optional): Number of input channels. Defaults to 2.
        hidden_channels (int, optional): Number of hidden state channels. Defaults to 32.
    """
    
    def __init__(self, input_channels=2, hidden_channels=32):
        # Initialize parent class
        super().__init__()
        
        # Create ConvLSTM cell for spatial-temporal processing
        self.cell = ConvLSTMCell(input_channels, hidden_channels)
        
        # Output projection layer: reduces hidden channels to single output channel
        self.output_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x, hidden=None):
        """
        Forward pass through the ConvLSTM network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            hidden (tuple, optional): Previous hidden and cell states (h, c). If None,
                                    zero states will be initialized.
                                    
        Returns:
            tuple: (output, (h, c)) where:
                - output: Per-pixel predictions of shape (batch_size, 1, height, width)
                - (h, c): Updated hidden and cell states for next time step
        """
        
        # Extract input dimensions
        B, C, H, W = x.shape

        # Initialize hidden and cell states if not provided
        if hidden is None:
            h = torch.zeros(B, 32, H, W, device=x.device)
            c = torch.zeros(B, 32, H, W, device=x.device)
        else:
            h, c = hidden

        # Process input through ConvLSTM cell
        h, c = self.cell(x, h, c)
        
        # Project hidden state to output predictions
        output = self.output_conv(h)
        
        # Apply sigmoid for per-pixel probability predictions
        output = torch.sigmoid(output)
        
        # Return predictions and updated states
        return output, (h, c)