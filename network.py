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


def one_hot(y, n_classes=OUTPUT_SIZE):
    """Turns integer labels (e.g. 7) into one-hot vectors (e.g. [0,0,0,0,0,0,0,1,0,0])."""
    encoded = np.zeros((y.size, n_classes))
    encoded[np.arange(y.size), y] = 1
    return encoded


def cross_entropy_loss(probs, y):
    """How wrong the predictions are, as a single number - lower is better.

    For each sample, this only looks at the probability the network assigned
    to the *correct* class and takes -log() of it. If the network was 100%
    confident in the right answer, that probability is 1 and -log(1) = 0 -
    zero loss, nothing to improve. The closer that probability gets to 0
    (very confident in the *wrong* answer), the more -log() blows up towards
    infinity - a large penalty. Averaged across the batch.
    """
    n_samples = probs.shape[0]
    correct_class_probs = probs[np.arange(n_samples), y]
    # Clip to avoid log(0), which would be -infinity.
    correct_class_probs = np.clip(correct_class_probs, 1e-12, 1.0)
    return -np.mean(np.log(correct_class_probs))


def backward(X, y, W2, cache):
    """Backpropagation: work out how much each weight contributed to the loss.

    This is the chain rule in action: the loss depends on A2, which depends
    on Z2, which depends on W2 and A1, which depends on Z1, which depends on
    W1. To find "how does changing W1 affect the loss" (all the way at the
    other end of that chain), we multiply the local effect of each step
    together, working backward from the loss to the first layer - hence
    "back"-propagation. Each line below is one link in that chain.
    """
    n_samples = X.shape[0]
    A1, A2, Z1 = cache["A1"], cache["A2"], cache["Z1"]
    Y = one_hot(y)

    # For softmax + cross-entropy together, this combined derivative
    # simplifies beautifully to just "predicted probabilities minus the
    # true one-hot labels" - a well-known result that avoids a much messier
    # separate derivative for softmax and for the loss.
    dZ2 = (A2 - Y) / n_samples

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)  # derivative of ReLU: 1 where Z1 was positive, 0 otherwise

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return dW1, db1, dW2, db2
