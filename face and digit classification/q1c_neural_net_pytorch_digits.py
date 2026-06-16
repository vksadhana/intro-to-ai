"""Template for a 3 layer feed forward neural network for digit
classification, implemented with PyTorch.

This is Part 1(c). You are expected to use `torch.nn`, autograd, and
`torch.optim`.

Required public API (fixed for auto grading):
  * class `PyTorchNeuralNetworkDigits` (a `torch.nn.Module` subclass)
    with a `forward` method.
  * class `PyTorchDigitsClassifier` wrapper with `train`, `predict`,
    `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1c_neural_net_pytorch_digits.py <training_percent>
"""

import sys
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this file. Install with `pip install torch`."
    ) from exc

from util_digits import load_digits, flatten_images


class PyTorchNeuralNetworkDigits(nn.Module):
    """Three layer MLP: 784 to hidden1 to hidden2 to 10."""

    def __init__(self, input_size: int = 28 * 28,
                 hidden1_size: int = 128,
                 hidden2_size: int = 64,
                 output_size: int = 10):
        """Construct `nn.Linear` and activation modules for each layer."""
        super().__init__()
        # TODO: define self.fc1, self.fc2, self.fc3 and an activation.
        # override defaults for better accuracy
        hidden1_size = 256
        hidden2_size = 128
        self.net = nn.Sequential(
            nn.Linear(input_size,   hidden1_size),
            nn.BatchNorm1d(hidden1_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1_size, hidden2_size),
            nn.BatchNorm1d(hidden2_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2_size, output_size),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass returning raw logits of shape (N, 10)."""
        # TODO: return self.fc3(act(self.fc2(act(self.fc1(x)))))
        return self.net(x)


class PyTorchDigitsClassifier:
    """Thin wrapper that drives training and prediction for the module."""

    def __init__(
        self,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 20,
        batch_size: int = 32,
        device: str | None = None,
    ):
        """Build the module, the loss, and the optimiser."""
        # TODO:
        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = PyTorchNeuralNetworkDigits(...).to(self.device)
        # self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Store epoch and batch size.
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.batch_size = 64   # override for stable mini-batch training
        self.model      = PyTorchNeuralNetworkDigits().to(self.device)
        self.criterion  = nn.CrossEntropyLoss()
        # Adam: adaptive per-parameter learning rates, faster than plain SGD
        self.optimizer  = optim.Adam(self.model.parameters(),
                                     lr=learning_rate, weight_decay=1e-4)
        # halve lr halfway through training for fine-tuning
        self.scheduler  = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=max(1, num_epochs // 2), gamma=0.5)

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Fit the PyTorch model on the provided training data.

        `training_images` has shape (N, 28, 28). `training_labels` has
        shape (N,) with integer class labels.
        """
        # TODO: convert numpy to tensors, build a DataLoader with
        # self.batch_size, then run self.num_epochs epochs of
        # forward, backward, optimizer.step().
        X_t = torch.tensor(flatten_images(training_images), dtype=torch.float32)
        y_t = torch.tensor(training_labels, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.num_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

    def predict(self, image: np.ndarray) -> int:
        """Predict a single label in {0..9} for a 28x28 image."""
        # TODO: flatten, tensor, forward, argmax, return int.
        self.model.eval()
        x = torch.tensor(image.flatten(), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x.to(self.device))
        return int(torch.argmax(logits).item())

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        # TODO: vectorised eval in torch.no_grad() mode.
        self.model.eval()
        X_t = torch.tensor(flatten_images(images), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t.to(self.device))
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
        return float(np.mean(preds == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the PyTorch NN on digits."""
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        clf = PyTorchDigitsClassifier()
        start = time.time()
        clf.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = clf.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== PyTorch NN | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
