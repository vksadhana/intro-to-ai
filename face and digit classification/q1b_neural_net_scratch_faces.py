"""Template for a 3 layer feed forward neural network for binary face
classification, implemented from scratch.

Architecture: input -> hidden1 -> hidden2 -> output.

Implement the forward pass, back propagation, and weight update
yourself. You may use numpy. You may not use torch, tensorflow,
sklearn, jax, or keras for training, gradients, or prediction.

Required public API (fixed for auto grading):
  * class `ScratchNeuralNetworkFaces` with methods `forward`,
    `backward`, `update_weights`, `train`, `predict`, `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1b_neural_net_scratch_faces.py <training_percent>
"""

import sys
import time
import numpy as np

from util_faces import load_faces, flatten_images


class ScratchNeuralNetworkFaces:
    """3 layer fully connected network for binary face detection:

    input (70*60 = 4200) -> hidden1 -> hidden2 -> output (1 or 2).

    You may model the output as a single sigmoid unit (binary cross
    entropy) or as two softmax units. Either is fine; just be
    consistent in `forward`, `backward`, and `predict`.
    """

    def __init__(
        self,
        input_size: int = 70 * 60,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        output_size: int = 2,
        learning_rate: float = 0.01,
        num_epochs: int = 20,
        batch_size: int = 32,
        seed: int | None = None,
    ):
        """Initialise network hyperparameters and weight matrices."""
        # TODO: initialize weights, biases, and hyperparameters.
        if seed is not None:
            np.random.seed(seed)

        # override defaults for better accuracy
        hidden1_size = 256
        hidden2_size = 128
        learning_rate = 0.02
        num_epochs = 25
        batch_size = 16

        self.lr         = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_out      = output_size
        self.momentum   = 0.9

        # He initialisation for ReLU networks
        def he(fan_in, fan_out):
            return (np.random.randn(fan_in, fan_out)
                    * np.sqrt(2.0 / fan_in)).astype(np.float32)

        self.W1 = he(input_size,   hidden1_size)
        self.b1 = np.zeros(hidden1_size, dtype=np.float32)
        self.W2 = he(hidden1_size, hidden2_size)
        self.b2 = np.zeros(hidden2_size, dtype=np.float32)
        self.W3 = he(hidden2_size, output_size)
        self.b3 = np.zeros(output_size,  dtype=np.float32)

        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3); self.vb3 = np.zeros_like(self.b3)

        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. See `ScratchNeuralNetworkDigits.forward`."""
        # TODO: 3 linear layers with hidden activations; final softmax
        # (or sigmoid if output_size == 1). Cache intermediates.
        Z1 = X  @ self.W1 + self.b1
        A1 = np.maximum(0.0, Z1)           # ReLU
        Z2 = A1 @ self.W2 + self.b2
        A2 = np.maximum(0.0, Z2)           # ReLU
        Z3 = A2 @ self.W3 + self.b3
        Z3 = Z3 - Z3.max(axis=1, keepdims=True)
        exp_z3 = np.exp(Z3)
        Y_hat = exp_z3 / exp_z3.sum(axis=1, keepdims=True)
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Y_hat": Y_hat}
        return Y_hat

    def backward(self, X: np.ndarray, y_onehot: np.ndarray) -> dict:
        """Back propagate loss gradients through the network."""
        # TODO: compute dW1, db1, dW2, db2, dW3, db3 via the chain rule.
        A1, A2, Y_hat = self.cache["A1"], self.cache["A2"], self.cache["Y_hat"]
        Z1, Z2        = self.cache["Z1"],  self.cache["Z2"]
        B = X.shape[0]

        d3  = (Y_hat - y_onehot) / B
        dW3 = A2.T @ d3;  db3 = d3.sum(axis=0)
        d2  = (d3 @ self.W3.T) * (Z2 > 0).astype(np.float32)
        dW2 = A1.T @ d2;  db2 = d2.sum(axis=0)
        d1  = (d2 @ self.W2.T) * (Z1 > 0).astype(np.float32)
        dW1 = X.T  @ d1;  db1 = d1.sum(axis=0)

        return {"dW1": dW1, "db1": db1,
                "dW2": dW2, "db2": db2,
                "dW3": dW3, "db3": db3}

    def update_weights(self, grads: dict) -> None:
        """Apply one gradient descent step using `grads` from `backward`."""
        # TODO: SGD update with self.learning_rate.
        self.vW3 = self.momentum*self.vW3 - self.lr*grads["dW3"]; self.W3 += self.vW3
        self.vb3 = self.momentum*self.vb3 - self.lr*grads["db3"]; self.b3 += self.vb3
        self.vW2 = self.momentum*self.vW2 - self.lr*grads["dW2"]; self.W2 += self.vW2
        self.vb2 = self.momentum*self.vb2 - self.lr*grads["db2"]; self.b2 += self.vb2
        self.vW1 = self.momentum*self.vW1 - self.lr*grads["dW1"]; self.W1 += self.vW1
        self.vb1 = self.momentum*self.vb1 - self.lr*grads["db1"]; self.b1 += self.vb1

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Full training loop: epochs and mini batches."""
        # TODO: flatten, one hot encode labels, epoch and mini batch loop.
        X = flatten_images(training_images).astype(np.float32)
        y = training_labels
        n = len(X)

        for _ in range(self.num_epochs):
            order = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                bi  = order[start:start + self.batch_size]
                Xb  = X[bi];  yb = y[bi]
                Yoh = np.zeros((len(bi), self.n_out), dtype=np.float32)
                Yoh[np.arange(len(bi)), yb] = 1.0
                self.forward(Xb)
                grads = self.backward(Xb, Yoh)
                self.update_weights(grads)

    def predict(self, image: np.ndarray) -> int:
        """Predict 0 or 1 for a single 70x60 image."""
        # TODO: flatten, run forward, argmax (or threshold).
        x = image.flatten().astype(np.float32).reshape(1, -1)
        probs = self.forward(x)
        return int(np.argmax(probs))

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        # TODO: vectorised forward pass, argmax, compare.
        X     = flatten_images(images).astype(np.float32)
        probs = self.forward(X)
        preds = np.argmax(probs, axis=1)
        return float(np.mean(preds == labels))


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the scratch NN on faces."""
    training_images, training_labels = load_faces("train")
    test_images, test_labels = load_faces("test")

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = training_images[idx]
        y_sample = training_labels[idx]

        net = ScratchNeuralNetworkFaces()
        start = time.time()
        net.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = net.evaluate(test_images, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== Scratch NN | Faces | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)
