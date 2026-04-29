import multiprocessing
import os
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary

import Config


device = "cuda" if torch.cuda.is_available() else "cpu"


class TinyVGG(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


def main():
    data_path = Path("C:/Users/anchi/GitHub/Repos/Image_Classification/Data_Refined")
    train_dir = data_path / "Train"
    test_dir = data_path / "Test"

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=data_transform,
                                      target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)

    class_names = train_data.classes
    print(f"Class names: {class_names}")
    print(f"Class to index mapping: {train_data.class_to_idx}")

    batch_size = 32
    # On Windows, keep workers at 0 for maximum stability with scripts.
    num_workers = 0 if os.name == "nt" else (os.cpu_count() or 0)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    model_0 = TinyVGG(input_shape=3,
                      hidden_units=10,
                      output_shape=len(train_data.classes)).to(device)
    print(model_0)
    summary(model_0, input_size=[1, 3, 64, 64])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Config helpers expect a module-level device variable.
    Config.device = device

    num_epochs = 5
    start_time = timer()
    model_0_results = Config.train(model=model_0,
                                   train_dataloader=train_dataloader,
                                   test_dataloader=test_dataloader,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   epochs=num_epochs)
    end_time = timer()

    print(f"Total training time: {end_time - start_time:.3f} seconds")
    return model_0_results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    model_0_results = main()
    Config.plot_loss_curves(model_0_results)
