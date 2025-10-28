import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell implementation.
    
    A single cell of ConvLSTM that processes spatial-temporal data by combining
    convolutional operations with LSTM gating mechanisms. This allows the model
    to capture both spatial features and temporal dependencies in sequential data.
    
    Args:
        input_channels (int): Number of channels in the input tensor
        hidden_channels (int): Number of channels in the hidden state
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
    """
    
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        
        # Single convolution layer that generates all four gates simultaneously
        # Input: concatenated input and hidden state
        # Output: 4 * hidden_channels (for input, forget, output, and candidate gates)
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 
            4 * hidden_channels, 
            kernel_size, 
            padding=padding
        )

    def forward(self, x, h, c):
        """
        Forward pass of the ConvLSTM cell.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            h (torch.Tensor): Hidden state of shape (batch_size, hidden_channels, height, width)
            c (torch.Tensor): Cell state of shape (batch_size, hidden_channels, height, width)
            
        Returns:
            tuple: (h_next, c_next) - Updated hidden and cell states
        """
        
        # Concatenate input and previous hidden state along channel dimension
        combined = torch.cat([x, h], dim=1) 
        
        # Generate all gates through a single convolution
        gates = self.conv(combined)
        
        # Split the output into four gate components
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        # Apply appropriate activations to each gate
        i = torch.sigmoid(i)  # Input gate: controls what information to store
        f = torch.sigmoid(f)  # Forget gate: controls what information to discard
        o = torch.sigmoid(o)  # Output gate: controls what parts of cell state to output
        g = torch.tanh(g)     # Candidate values: new information that could be stored
        
        # Update cell state: forget old information and add new information
        c_next = f * c + i * g
        
        # Update hidden state: filter cell state through output gate
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
