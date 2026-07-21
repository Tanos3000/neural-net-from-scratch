"""The neural network itself: parameter init, activations, and forward pass.

Everything here is plain NumPy - matrix multiplication and a couple of
elementwise functions. No autodiff, no framework magic.
"""

import numpy as np

INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10


def init_params(seed: int = 42):
    """Random starting weights and biases.

    Weights start small and random, not zero: if every weight started at the
    same value (e.g. zero), every hidden neuron would compute the exact same
    thing during the forward pass and get the exact same update during
    training - they'd never become different from each other. Random values
    break that symmetry so each neuron can learn something different.

    The scale (multiplying by sqrt(2 / fan_in)) is a common rule of thumb
    ("He initialization") for networks using ReLU - it keeps the numbers
    flowing through the network from shrinking to zero or blowing up as they
    pass through each layer, which matters more as networks get deeper.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((INPUT_SIZE, HIDDEN_SIZE)) * np.sqrt(2 / INPUT_SIZE)
    b1 = np.zeros(HIDDEN_SIZE)
    W2 = rng.standard_normal((HIDDEN_SIZE, OUTPUT_SIZE)) * np.sqrt(2 / HIDDEN_SIZE)
    b2 = np.zeros(OUTPUT_SIZE)
    return W1, b1, W2, b2


def relu(Z):
    """ReLU: 'Rectified Linear Unit'. output = Z if Z > 0, else 0.

    The simplest common activation function. Without any nonlinear activation
    function at all, stacking multiple layers would be mathematically no
    different from having just one layer - a chain of linear operations
    collapses into a single linear operation. ReLU's simple nonlinearity is
    what lets multiple layers actually learn more than one layer could.
    """
    return np.maximum(0, Z)


def softmax(Z):
    """Turns raw output scores into probabilities that sum to 1 across classes.

    Z has shape (n_samples, 10) - one raw score per digit class. Softmax
    exponentiates each score (so everything becomes positive) and divides by
    the row sum (so each row of probabilities adds up to 1). Subtracting the
    row max before exponentiating doesn't change the result mathematically,
    but avoids computing exp() of large numbers, which would overflow.
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    """One forward pass: input images -> hidden layer -> output probabilities.

    Returns every intermediate value (not just the final prediction) because
    backpropagation (Step 3) needs them to compute gradients.
    """
    Z1 = X @ W1 + b1       # weighted sum into the hidden layer
    A1 = relu(Z1)           # hidden layer activations
    Z2 = A1 @ W2 + b2       # weighted sum into the output layer
    A2 = softmax(Z2)        # output probabilities (one per digit, per sample)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache
