"""
Trains a Pytorch Image classification model using device agnostic code.
"""
import os
import torch
from torch import nn
import data_setup, engine, model_builder, utils
from torchvision import transforms
import argparse

def main(args):
    train_dir = args.train_dir
    test_dir = args.test_dir

    # Set the device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Disable cuDNN benchmarking
    torch.backends.cudnn.benchmark = False
    # Enable deterministic algorithms
    torch.backends.cudnn.deterministic = True

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=args.batch_size
    )

    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=args.hidden_units,
        output_shape=len(class_names)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.num_epochs,
        device=device
    )

    utils.save_model(
        model=model,
        target_dir=args.model_dir,
        model_name=args.model_name
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Only necessary for freezing the script

    parser = argparse.ArgumentParser(description="Train a PyTorch image classification model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units in the model')
    parser.add_argument('--train_dir', type=str, default="data/pizza_steak_sushi/train", help='Directory for training data')
    parser.add_argument('--test_dir', type=str, default="data/pizza_steak_sushi/test", help='Directory for testing data')
    parser.add_argument('--model_dir', type=str, default="models", help='Directory to save the trained model')
    parser.add_argument('--model_name', type=str, default="05_going_modular_script_mode_tinyvgg_model.pth", help='Name of the saved model')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')

    args = parser.parse_args()
    main(args)
