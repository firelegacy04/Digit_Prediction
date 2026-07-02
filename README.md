# MNIST Digit Classification via Custom NumPy Neural Network

https://colornumbers-bgraqqriwknfgppodyrrcv.streamlit.app/ 

## Overview
This repository contains a complete, scratch-built Artificial Neural Network (ANN) designed to classify the MNIST handwritten digit dataset. The primary objective of this project is to implement deep learning fundamentals—specifically forward propagation, backpropagation, and gradient descent—without relying on high-level machine learning frameworks such as TensorFlow or PyTorch.

## Mathematical Implementation
The network is implemented entirely in Python using the NumPy library for matrix operations. The learning process relies on the following mathematical components:

* **Forward Propagation:** Matrix multiplication of input vectors with weight matrices, followed by the addition of bias vectors.
* **Activation Functions:** * **ReLU (Rectified Linear Unit):** Applied to the hidden layer to introduce non-linearity.
    * **Softmax:** Applied to the output layer to normalize raw logits into a valid probability distribution.
* **Backpropagation:** The network calculates the gradient of the loss function with respect to the weights using the chain rule. Weights are updated iteratively via Full Batch Gradient Descent.

## Network Architecture
The model follows a standard feedforward multi-layer perceptron (MLP) structure:
* **Input Layer:** 784 neurons, corresponding to the flattened 28x28 pixel input image.
* **Hidden Layer:** 128 neurons.
* **Output Layer:** 10 neurons, representing classes 0 through 9.

**Performance:** The model achieves approximately 93.25% accuracy on the standard MNIST test set after 500 training iterations.

## Inference and Image Preprocessing
The project includes a Streamlit-based graphical user interface that allows users to draw digits on a canvas for real-time inference.

To bridge the gap between raw user input and the model's expected data distribution, the application mathematically replicates the original MNIST dataset preprocessing pipeline. Upon receiving a user's drawn input, the application:
1. Isolates the strict bounding box of the non-zero pixels.
2. Scales the bounding box such that the maximum dimension is exactly 20 pixels, utilizing Lanczos resampling to preserve data integrity.
3. Calculates the required offset and translates the scaled digit onto the center of a 28x28 zero-matrix.

This precise geometric alignment ensures that the spatial distribution of the real-time input matches the distribution of the original training data, yielding robust and accurate real-world predictions.
