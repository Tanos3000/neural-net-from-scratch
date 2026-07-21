"""The training loop: repeatedly apply gradient descent until the network
actually learns something.
"""

import numpy as np

from network import forward, backward, cross_entropy_loss


def accuracy(probs, y):
    predictions = probs.argmax(axis=1)
    return (predictions == y).mean()


def train(X_train, y_train, W1, b1, W2, b2, epochs=10, batch_size=64, learning_rate=0.5, seed=0):
    """Mini-batch gradient descent.

    One epoch = one full pass through the training data, split into small
    batches. For each batch: forward pass -> loss -> backward pass -> nudge
    every weight a small step in the direction that reduces the loss. Repeat
    for every batch, then repeat the whole thing for `epochs` passes.

    Why mini-batches instead of the whole dataset at once? Computing exact
    gradients over all 60,000 images before taking a single update step would
    be slow and only update the weights once per epoch. Small batches (here,
    64 images at a time) give a noisier but much more frequent gradient
    estimate, so the network gets far more chances to improve per epoch -
    the standard tradeoff in practice.
    """
    rng = np.random.default_rng(seed)
    n_samples = X_train.shape[0]
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        # Shuffle each epoch so the network doesn't see batches in the same
        # order every time, which could otherwise bias training.
        permutation = rng.permutation(n_samples)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        epoch_losses = []
        for start in range(0, n_samples, batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]

            probs, cache = forward(X_batch, W1, b1, W2, b2)
            loss = cross_entropy_loss(probs, y_batch)
            epoch_losses.append(loss)

            dW1, db1, dW2, db2 = backward(X_batch, y_batch, W2, cache)

            # Gradient descent update: the gradient points in the direction
            # that *increases* the loss, so we step in the opposite
            # direction (subtract), scaled by the learning rate.
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Evaluate on the full training set at the end of each epoch to
        # track progress (not used to update weights, just to monitor).
        train_probs, _ = forward(X_train, W1, b1, W2, b2)
        epoch_accuracy = accuracy(train_probs, y_train)
        mean_loss = np.mean(epoch_losses)

        history["loss"].append(mean_loss)
        history["accuracy"].append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {mean_loss:.4f} - train accuracy: {epoch_accuracy*100:.1f}%")

    return W1, b1, W2, b2, history
