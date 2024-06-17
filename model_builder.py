"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self,
               input_shape : int,
               hidden_units : int,
               output_shape : int):
    super().__init__()
    self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels = input_shape,
                  out_channels = hidden_units,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units,
                  out_channels = hidden_units,
                  kernel_size =3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2,
                     stride = 2),
        nn.Dropout(0.3)
    )
    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels = hidden_units,
                  out_channels = hidden_units*2,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(hidden_units*2),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_units*2,
                  out_channels = hidden_units*2,
                  kernel_size = 3,
                  stride = 1,
                  padding = 1),
        nn.BatchNorm2d(hidden_units*2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2,
                  stride = 2),
        nn.Dropout(0.3)
    )
    self.conv_block3 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units*2,
                  out_channels=hidden_units*4,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.BatchNorm2d(hidden_units*4),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units*4,
                  out_channels=hidden_units*4,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        #nn.BatchNorm2d(hidden_units*4),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.4)  # Slightly increasing dropout for deeper layers
    )
    self.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(in_features = hidden_units*4,
                  out_features = output_shape)
    )
  def forward(self, x: torch.Tensor):
    return self.classifier(self.conv_block3(self.conv_block2(self.conv_block1(x))))
