### Task 1 postman
 
# Feedforward Neural Network with Manual Backpropagation

A small feedforward neural network implemented from scratch in NumPy — including the forward pass, manual backpropagation (no autograd/`.backward()`), and gradient descent training — to classify handwritten digits from the `sklearn` `digits` dataset.

## Project Structure
 NeuralNetwork.py      
 test_gradients.py    
 WRITEUP.md            
 README.md             


## Architecture


Input (64) -> Linear (W1, b1) -> ReLU -> Linear (W2, b2) -> Softmax -> Output (10)


- **Dataset:** `sklearn.datasets.load_digits()` — 1,797 images, 8x8 pixels (64 features), 10 classes (digits 0–9).
- **Hidden layer size:** 32.
- **Loss:** Cross-entropy loss.
- **Optimizer:** Plain gradient descent (learning rate 0.1).

## Requirements

- Python 3.x
- NumPy
- scikit-learn


## How to Run

### 1. Train the network

Run the main script to train the network for 1,000 epochs. Loss is printed every 100 epochs.


python NeuralNetwork.py


Expected output: the loss should start around `~2.3` (consistent with random, untrained weights on a 10-class problem) and decrease steadily to well under `0.05` by the end of training.

### 2. Run the correctness harness (gradient check)

This script compares the manually-derived backpropagation gradients against an independently computed numerical gradient (via finite differences), and reports PASS/FAIL for each parameter.

```
python test_gradients.py
```

Expected output: `PASS` for `dW1`, `db1`, `dW2`, `db2`, each with a very small max difference (on the order of `1e-10` or smaller), confirming the manual gradients are correct.

## Notes

- The backward pass is implemented entirely manually using the chain rule — no `torch.autograd`, `.backward()`, or other automatic differentiation is used in the core network implementation. `torch`/numerical gradient checking is used only in `test_gradients.py`, purely to *verify* correctness, not to compute training gradients.
- See `WRITEUP.md` for the full mathematical derivation, training results, gradient-check results, and a discussion of mistakes made (and common pitfalls in this type of implementation) along the way.

