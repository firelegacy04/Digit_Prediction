import numpy as np

# ==========================================
# 1. PARSE THE LOCAL UNCOMPRESSED MNIST DATA
# ==========================================
def load_mnist():
    """Loads the MNIST dataset from local uncompressed binary files."""
    # Updated filenames to match your PyCharm directory exactly
    files = {
        "train_images": "train-images.idx3-ubyte",
        "train_labels": "train-labels.idx1-ubyte",
        "t10k_images": "t10k-images.idx3-ubyte",
        "t10k_labels": "t10k-labels.idx1-ubyte"
    }

    data = {}
    for key, filename in files.items():
        # Changed gzip.open to standard open
        with open(filename, 'rb') as f:
            # Skip the magic number
            f.read(4)
            num_items = int.from_bytes(f.read(4), 'big')

            if "images" in key:
                rows = int.from_bytes(f.read(4), 'big')
                cols = int.from_bytes(f.read(4), 'big')
                # Read the rest of the buffer into a numpy array and reshape it
                data[key] = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows * cols)
            else:
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)

    return (data['train_images'], data['train_labels']), (data['t10k_images'], data['t10k_labels'])

# ==========================================
# 2. MATH & ACTIVATION FUNCTIONS
# ==========================================
def relu(Z):
    """Rectified Linear Unit: Returns Z if Z > 0, else 0."""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Derivative of ReLU: Returns 1 if Z > 0, else 0."""
    return Z > 0

def softmax(Z):
    """Converts raw scores into probabilities that sum to 1."""
    A = np.exp(Z - np.max(Z, axis=0)) # Subtracted max for numerical stability
    return A / np.sum(A, axis=0)

def one_hot(Y):
    """Converts a 1D array of labels (0-9) into a 2D matrix of binary vectors."""
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# ==========================================
# 3. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
def init_params():
    """Initializes weights and biases for a 784 -> 128 -> 10 network."""
    W1 = np.random.randn(128, 784) * np.sqrt(2. / 784)
    b1 = np.zeros((128, 1))

    W2 = np.random.randn(10, 128) * np.sqrt(2. / 128)
    b2 = np.zeros((10, 1))

    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    """Pushes data forward through the network."""
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """Calculates gradients (how much to adjust weights) by working backward."""
    m = Y.size
    one_hot_Y = one_hot(Y)

    # Calculate error at the output layer
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    # Propagate error back to the hidden layer
    dZ1 = W2.T.dot(dZ2) * relu_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """Updates the weights and biases based on the calculated gradients."""
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# ==========================================
# 4. TRAINING LOOP & EVALUATION
# ==========================================
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def train_network(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        # 1. Forward pass
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        # 2. Backward pass
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)

        # 3. Update parameters
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # Print progress
        if i % 50 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print(f"Iteration: {i:3} | Accuracy: {accuracy * 100:.2f}%")

    return W1, b1, W2, b2

# ==========================================
# 5. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    print("--- 1. Loading Local Dataset ---")
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # Transpose (.T) and normalize pixel values to 0-1
    X_train = x_train.T / 255.0
    Y_train = y_train

    X_test = x_test.T / 255.0
    Y_test = y_test

    print("\n--- 2. Starting Training ---")
    W1, b1, W2, b2 = train_network(X_train, Y_train, alpha=0.15, iterations=500)

    print("\n--- 3. Evaluating on Test Data ---")
    _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
    test_predictions = get_predictions(A2_test)
    test_accuracy = get_accuracy(test_predictions, Y_test)

    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    print("\n--- 4. Saving Model Weights ---")
    np.savez("mnist_weights.npz", W1=W1, b1=b1, W2=W2, b2=b2)
    print("Weights saved successfully to 'mnist_weights.npz'!")