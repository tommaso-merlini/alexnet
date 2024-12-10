# AlexNet-CIFAR10

A PyTorch implementation of a modified AlexNet architecture optimized for the CIFAR-10 dataset. This project adapts the revolutionary AlexNet architecture (Krizhevsky et al., 2012) to work with smaller 32x32 images, making it suitable for CIFAR-10 classification tasks.

## Overview

This implementation features a memory-efficient version of AlexNet, specifically designed for the CIFAR-10 dataset. The architecture has been modified to handle 32x32 pixel images while maintaining the core concepts of the original paper.

## Architecture

The model architecture includes:
```python
Model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    
    nn.Flatten(),
    nn.Linear(96 * 4 * 4, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
)
```

## Dataset

The CIFAR-10 dataset consists of:
- 60,000 32x32 color images
- 10 different classes
- 50,000 training images
- 10,000 test images

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib (for visualization)

<!-- ## Installation -->
<!---->
<!-- ```bash -->
<!-- git clone https://github.com/yourusername/AlexNet-CIFAR10.git -->
<!-- cd AlexNet-CIFAR10 -->
<!-- pip install -r requirements.txt -->
<!-- ``` -->

## Usage

```python
# Train and Test the model
python main.py
```

## Key Modifications from Original AlexNet

1. Reduced number of convolutional layers
2. Smaller kernel sizes
3. Added MaxPooling layers
4. Memory-efficient architecture
5. Adapted for 32x32 input size

## License

MIT

## Acknowledgments

- Original AlexNet paper: "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al.
- CIFAR-10 dataset: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
