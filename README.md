1. Download the Dataset

Ensure the original uncompressed MNIST dataset files are placed in the root directory of this project alongside your python scripts.
2. Train the Neural Network

Before running the app, you need to train the model so it can learn how to recognize digits.
Bash

python neural_network.py

Note: This will process 60,000 images over 500 iterations. It may take a couple of minutes depending on your CPU. Once finished, it will save a mnist_weights.npz file to your folder.
3. Launch the Web App

Once the weights file is generated, you can start the interactive Streamlit interface:
Bash

streamlit run streamlit.py

A browser window will open automatically. Navigate to the "Play Area" tab, draw a digit, and watch the math work in real-time!
🧠 Neural Network Architecture

    Input Layer: 784 nodes (for 28x28 pixel flattened images)

    Hidden Layer: 128 nodes with a ReLU (Rectified Linear Unit) activation function

    Output Layer: 10 nodes (for digits 0-9) with a Softmax activation function to generate confidence probabilities
    """
