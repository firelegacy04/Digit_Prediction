import numpy as np
from scipy.ndimage import shift, zoom
import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. DATA AUGMENTATION (Kept mostly identical)
# ==========================================
def augment_image(x):
    """
    x: (784,) MNIST image (flat)
    returns: (784,) augmented image
    """
    img = x.reshape(28, 28)

    # 1. Random shift
    dx, dy = np.random.randint(-2, 3), np.random.randint(-2, 3)
    img = shift(img, shift=[dx, dy], mode='constant', cval=0)

    # 2. Random zoom
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        zoomed = zoom(img, zoom=scale)
        canvas = np.zeros((28, 28))

        h, w = zoomed.shape
        h_start = max((28 - h) // 2, 0)
        w_start = max((28 - w) // 2, 0)

        zoomed = zoomed[:28, :28]  # safety crop
        canvas[:zoomed.shape[0], :zoomed.shape[1]] = zoomed
        img = canvas

    return img.reshape(-1)


# ==========================================
# 2. PARSE THE LOCAL UNCOMPRESSED MNIST DATA
# ==========================================
def load_mnist():
    """Loads the MNIST dataset from local uncompressed binary files."""
    files = {
        "train_images": "train-images.idx3-ubyte",
        "train_labels": "train-labels.idx1-ubyte",
        "t10k_images": "t10k-images.idx3-ubyte",
        "t10k_labels": "t10k-labels.idx1-ubyte"
    }

    data = {}
    for key, filename in files.items():
        with open(filename, 'rb') as f:
            f.read(4)  # Skip magic number
            num_items = int.from_bytes(f.read(4), 'big')

            if "images" in key:
                rows = int.from_bytes(f.read(4), 'big')
                cols = int.from_bytes(f.read(4), 'big')
                data[key] = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows * cols)
            else:
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)

    return (data['train_images'], data['train_labels']), (data['t10k_images'], data['t10k_labels'])


# ==========================================
# 3. THE CNN ARCHITECTURE (PyTorch)
# ==========================================
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        # Convolution Layer: 1 input channel (grayscale), 16 output filters, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Max Pooling: reduces 28x28 down to 14x14
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer: 16 filters * 14x14 spatial dimensions
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape enters as: (Batch_Size, 1, 28, 28)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten for the dense layer
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # No softmax needed here; CrossEntropyLoss handles it internally
        return x


# ==========================================
# 4. TRAINING LOOP & EVALUATION
# ==========================================
def train_network():
    print("--- 1. Loading Local Dataset ---")
    (X_train, Y_train), (X_test, Y_test) = load_mnist()

    # Normalize to 0-1
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    model = SmallCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    batch_size = 64
    num_samples = X_train.shape[0]

    print("\n--- 2. Starting Training ---")
    for epoch in range(epochs):
        # Shuffle indices for the epoch
        indices = np.random.permutation(num_samples)
        running_loss = 0.0

        # Process in Mini-Batches (massively speeds up training)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]

            # Extract and augment the batch
            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]

            # Apply augmentation only to this tiny batch of 64 images, not all 60k
            X_batch_aug = np.array([augment_image(img) for img in X_batch])

            # Convert NumPy arrays to PyTorch Tensors and reshape to (Batch, Channels, H, W)
            inputs = torch.tensor(X_batch_aug, dtype=torch.float32).view(-1, 1, 28, 28)
            labels = torch.tensor(Y_batch, dtype=torch.long)

            # 1. Zero the gradients
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(inputs)

            # 3. Calculate Loss
            loss = criterion(outputs, labels)

            # 4. Backward pass (Backprop)
            loss.backward()

            # 5. Update weights
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {running_loss / (num_samples / batch_size):.4f}")

    # --- Evaluation ---
    print("\n--- 3. Evaluating on Test Data ---")
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        inputs_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
        labels_test = torch.tensor(Y_test, dtype=torch.long)

        outputs = model(inputs_test)
        _, predictions = torch.max(outputs, 1)
        accuracy = (predictions == labels_test).sum().item() / len(labels_test)

    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

    print("\n--- 4. Saving Model Weights ---")
    torch.save(model.state_dict(), "cnn_mnist.pth")
    print("Weights saved successfully to 'cnn_mnist.pth'!")


# ==========================================
# 5. EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    train_network()