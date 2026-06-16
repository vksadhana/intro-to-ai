# final project — face and digit classification

implements and compares three classifiers on two image datasets: handwritten digit recognition (0–9) and binary face detection from edge-detected images.

adapted from Berkeley's CS188 classification project.

## classifiers implemented

- **perceptron** — online linear classifier with weight updates on misclassified examples
- **3-layer neural network (from scratch)** — manual implementation of forward pass, backpropagation, and weight updates using NumPy; no ML libraries used for training logic
- **3-layer neural network (PyTorch)** — same architecture using PyTorch's autograd and optimizer

## experiments

all three classifiers are trained on 10%–100% of the training set (in 10% increments) and evaluated on:
- prediction accuracy on held-out test set
- training time as a function of dataset size
- standard deviation of prediction error across runs

results reported separately for digits and faces.

## datasets

- handwritten digits: 28×28 pixel images, labels 0–9
- face detection: edge-detected images, binary label (face / not face)
- data: [rl.cs.rutgers.edu/fall2019/data.zip](http://rl.cs.rutgers.edu/fall2019/data.zip)

## stack

- Python (NumPy, PyTorch)
- report typeset in LaTeX
