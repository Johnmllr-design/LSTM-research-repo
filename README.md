# LSTM Research Repository

A PyTorch implementation of Convolutional LSTM (ConvLSTM) for spatial-temporal data processing, designed for microbubble detection on 
B-mode adn C-mode images. The repository doesn't contain the underlying data and training loop out of respect for the proprietary nature
of the data not belonging to me, however, this repositiory exists to showcase the underlying LSTM architecture for data pipelining and parameter
tuning that I learned and subsequently built. 


## 🏗️ Architecture

### Model Components (Husk/kernel design patter)

- **[ConvLSTMCell](./convLSTMcell.py)** - Core ConvLSTM cell implementation with gating mechanisms
- **[ConvLSTM](./convLSTM.py)** - network wrapper 

### Architecture Diagram
![Model Architecture](./model_architecture.png)

## Research Documentation

📋 **[Research Report (PDF)](./Miller_John_Report.pdf)** - technical documentation and analysis

## 📊 Research Poster

![Research Poster](./poster.png)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LSTM-research-repo.git
cd LSTM-research-repo

# Install dependencies
pip install torch torchvision
```


### Basic Usage

```python
# Some pseudocode

# The key is to utilize the hidden and cell state to maintain and modify the
# amount of "memory" is carried into the next inference frame
for frame in sequence:
    output, (hidden, cell) = model(frame, hidden, cell)
    # Process output, and reuse the new hidden and cell state for the next iteration...
```
