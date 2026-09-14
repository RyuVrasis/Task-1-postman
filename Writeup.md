# Write-up: Feedforward Neural Network with Manual Backpropagation

## 1. Overview

This project implements a small feedforward neural network entirely from scratch using NumPy, without relying on any autograd engine or `.backward()` call. The network is trained on the `sklearn` `digits` dataset (1,797 images of handwritten digits, 8x8 pixels each, flattened to 64 input features, 10 output classes) to classify digits 0–9.

The network architecture is:

```
Input (64) -> Linear (W1, b1) -> ReLU -> Linear (W2, b2) -> Softmax -> Output (10)
```

with a hidden layer size of 32, chosen arbitrarily as a reasonable size for this small dataset.

All four required deliverables were completed:
1. Forward pass implementation
2. Manual backward pass (backpropagation) derivation and implementation
3. Gradient checking against numerical gradients (correctness harness)
4. Training the network, demonstrating decreasing loss

## 2. Forward Pass

The forward pass computes:

```
z1 = X @ W1 + b1
a1 = ReLU(z1)
z2 = a1 @ W2 + b2
a2 = softmax(z2)
```

- `ReLU(z) = max(0, z)` — a simple nonlinearity that zeroes out negative values and passes positive values unchanged. Without a nonlinearity like this between layers, stacking multiple linear layers would collapse mathematically into a single linear transformation, no matter how many layers were added.
- `softmax(z)` converts the final layer's raw scores into a probability distribution over the 10 digit classes, so the outputs sum to 1 and can be interpreted as class-membership probabilities.

The loss function used is **cross-entropy loss**, averaged over the batch (which functions as the cost function):

```
L = -mean(log(a2[correct class]))
```

This penalizes the network heavily when it is confident but wrong, and lightly when it is uncertain but the correct class still has reasonable probability mass.

## 3. Backward Pass Derivation

Backpropagation was derived manually using the chain rule, propagating the gradient of the loss backward through the network, layer by layer.

### 3.1 Output layer gradient

For the combination of softmax activation and cross-entropy loss, the gradient of the loss with respect to the pre-softmax output `z2` simplifies elegantly to:

```
dz2 = (a2 - Y_onehot) / n
```

where `Y_onehot` is the true label encoded as a one-hot vector (a vector with a 1 in the position of the correct class and 0 elsewhere), and `n` is the number of samples in the batch. This expression is simply "predicted probability minus true probability" — the raw prediction error — which is why the softmax + cross-entropy pairing is the standard choice for classification: the gradient reduces to something both mathematically clean and intuitively interpretable.

### 3.2 Output layer weight and bias gradients

Using the chain rule through the linear layer `z2 = a1 @ W2 + b2`:

```
dW2 = a1.T @ dz2
db2 = sum(dz2, axis=0)
```

`dW2` combines how active each hidden neuron was (`a1`) with how much error came out the other end (`dz2`). `db2` sums the error signal directly, since the bias contributes equally regardless of the input.

### 3.3 Propagating the gradient into the hidden layer

The error is sent backward through `W2` (using its transpose, since we are now moving in the reverse direction through the same connection):

```
da1 = dz2 @ W2.T
```

It must then be passed backward through the ReLU nonlinearity. Because ReLU's derivative is 1 wherever its input was positive and 0 wherever its input was negative (a flat, zero-slope region), the incoming gradient is masked accordingly:

```
dz1 = da1 * (a1 > 0)
```

### 3.4 Hidden layer weight and bias gradients

Following the same pattern as the output layer:

```
dW1 = X.T @ dz1
db1 = sum(dz1, axis=0)
```

### 3.5 Gradient descent update

Once all four gradients are computed, the weights are updated in the direction that reduces the loss:

```
W = W - learning_rate * dW
b = b - learning_rate * db
```

Because gradient descent *subtracts* the gradient, a positive gradient component (meaning "this weight is contributing to excess loss in this direction") decreases the corresponding weight, and a negative gradient component increases it — automatically pushing incorrect-class probabilities down and the correct-class probability up over many iterations.

## 4. Gradient Checking (Correctness Harness)

To verify the manually-derived gradients were correct, a separate script, `test_gradients.py`, implements **numerical gradient checking**. For each parameter, a tiny perturbation (`epsilon = 1e-5`) is applied in both directions, and the resulting change in loss is used to estimate the gradient directly from its definition:

```
numerical_gradient = (loss(param + epsilon) - loss(param - epsilon)) / (2 * epsilon)
```

This numerical estimate was compared against the analytically-computed gradients (`dW1`, `db1`, `dW2`, `db2`) from the backward pass. Results on a small batch of 5 samples:

| Parameter | Result | Max difference |
|---|---|---|
| dW1 | PASS | ~3.1 × 10⁻¹¹ |
| db1 | PASS | ~4.9 × 10⁻¹² |
| dW2 | PASS | ~1.1 × 10⁻¹⁰ |
| db2 | PASS | ~3.1 × 10⁻¹² |

All differences are at the level of floating-point precision noise, confirming that the manually derived and implemented backward pass is mathematically correct.

## 5. Training Results

The network was trained for 1,000 epochs on the full dataset using a learning rate of 0.1. The loss decreased consistently and smoothly throughout training, with no oscillation or divergence:

| Epoch | Loss |
|---|---|
| 0 | 2.304 |
| 100 | 0.224 |
| 200 | 0.063 |
| 300 | 0.038 |
| 400 | 0.025 |
| 500 | 0.017 |
| 600 | 0.013 |
| 700 | 0.010 |
| 800 | 0.008 |
| 900 | 0.007 |

The initial loss (~2.30) closely matches the theoretical expectation for an untrained 10-class classifier, `-log(1/10) ≈ 2.303`, since randomly initialized weights produce roughly uniform probabilities across all classes. The smooth, monotonic decrease in loss (rather than any instability) is itself further indirect evidence that the gradients being used for the weight updates are correct.

## 6. Mistakes Made and How They Were Found and Fixed

Several bugs were encountered during implementation, each instructive in its own way:

- **Typo: `np.zeroes` instead of `np.zeros`.** This produced an `AttributeError`, immediately caught because NumPy does not have a function by that name. Python's own error message suggested the correct spelling, making this a one-line fix.

- **Syntax error: an extra closing parenthesis** (`print(output[0]))`) caused a `SyntaxError` pointing directly at the offending character. Fixed by removing the extra `)`.

- **Case-sensitivity bugs: `X` vs `x`, and `W1`/`W2` vs `w1`/`w2`.** Python treats variable names as case-sensitive, so a data-loading line written as `x = digits.data` while the rest of the code referenced `X` caused a `NameError`. The same issue occurred with weight matrices initialized as lowercase `w1`/`w2` while later code referenced uppercase `W1`/`W2`. Both were resolved by standardizing on a single consistent casing convention (uppercase for matrices `X`, `W1`, `W2`, following common ML notation for 2D matrices, lowercase for vectors like `y`, `b1`, `b2`) throughout the file.

- **Duplicate/leftover code from incremental development.** While building the backward pass incrementally, an earlier, redundant `forward()` definition and duplicate gradient-computation blocks were left in the file outside the training loop. These didn't cause errors (later definitions simply overwrote earlier ones), but were a source of confusion when debugging. This was resolved by consolidating the file into a single clean version with one definition per function and the training loop as the only place gradients are computed and applied.

None of these were conceptual errors in the actual backpropagation math — they were entirely mechanical, arising from being new to Python's syntax and case-sensitivity rules. The gradient-checking harness (Section 4) was what gave real confidence that the underlying calculus, as opposed to the code mechanics, was correct.

### Common mistakes in this type of project (beyond what was personally encountered)

Manual backpropagation implementations are prone to a well-known set of conceptual pitfalls, worth noting even where they were avoided here:

- **Forgetting to divide by batch size `n`.** Omitting the `/n` term in `dz2` (or in the loss itself) doesn't break the direction of the gradient, but scales its magnitude by the batch size, effectively acting like an unintended, much larger learning rate. This can cause training to diverge or become unstable, especially with larger batches.

- **Transposing the wrong matrix, or in the wrong order.** Expressions like `a1.T @ dz2` versus `dz2 @ a1.T` (or forgetting `.T` entirely) produce a shape mismatch or, worse, a shape that happens to broadcast incorrectly without throwing an error — silently computing something numerically wrong rather than crashing. Checking output shapes against the corresponding weight matrix's shape at every step, as done in Section 3, is the most reliable way to catch this early.

- **Forgetting to apply the ReLU derivative mask.** A common error is propagating `da1` straight into `dW1` without multiplying by `(a1 > 0)` first, effectively treating the hidden layer as if it had no activation function at all. This produces a model that still runs and even trains to some degree, but does not correctly reflect the network that was actually defined, and any accuracy numbers obtained from it would not be trustworthy.

- **Omitting the max-subtraction stability trick in softmax.** Computing `exp(z)` directly without first subtracting `max(z)` can overflow for large input values, producing `NaN` outputs. This is easy to miss early on since it may not appear until training has pushed some weights to larger magnitudes.

- **Using the wrong `axis` in `np.sum` or `np.max` for bias gradients or softmax normalization.** Since the data here is organized as (samples, features), reducing along the wrong axis silently produces an output of the wrong shape or wrong meaning (e.g., summing across classes instead of across samples), which can go unnoticed if not explicitly shape-checked.

- **Skipping gradient checking altogether and trusting that "training runs" means "backprop is correct."** A network can still show a decreasing loss curve even with a partially incorrect gradient (for instance, if only one layer's gradient is wrong but the other layer can partially compensate). This is why an independent numerical gradient check (Section 4) is a meaningfully stronger form of verification than only observing the loss curve.

## 7. Conclusion

This project implements a feedforward neural network's forward pass, manual backpropagation, and gradient descent training entirely from first principles in NumPy. The manually-derived gradients were verified against numerical gradient checking and found to match to within floating-point precision, and the network successfully learned to classify handwritten digits, with loss decreasing from ~2.3 to ~0.007 over 1,000 training epochs.
