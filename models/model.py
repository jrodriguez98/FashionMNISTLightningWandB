import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input image
        x = self.sequential(x)  # Process the image

        return x  # Return the logits

