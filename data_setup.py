"""
Contains the functionality for creating PyTorch DataLoader
for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir : str,
    test_dir : str,
    transform : transforms.Compose,
    batch_size : int,
    num_workers : int = NUM_WORKERS):
  """
  Creates training and testing dataloaders.
  Takes in a training and testing directory path and turns them
  into PyTorch datasets and dataloaders.

  Args :
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns :
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
  """
  # Use ImageFolder to create dataset
  train_data = datasets.ImageFolder(root = train_dir,
                                    transform = transform)
  test_data = datasets.ImageFolder(root = test_dir,
                                   transform = transform)
  class_names = train_data.classes

  train_dataloader = DataLoader(dataset = train_data,
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers = NUM_WORKERS,
                                pin_memory = True,
                                )
  test_dataloader = DataLoader(dataset = test_data,
                               batch_size = batch_size,
                               shuffle = False,
                               num_workers = NUM_WORKERS,
                               pin_memory = True)

  return train_dataloader , test_dataloader , class_names
