# 🔢 MNIST Digit Classification via PyTorch Convolutional Neural Network (CNN)
🔗 **Live Interactive Demo:** [Try the Streamlit App Here]("https://firelegacy04-digit-prediction-main-yjv0u7.streamlit.app/")

## 📝 Overview
This repository contains an interactive, web-deployed deep learning application designed to classify handwritten digits. Originally built as a scratch NumPy Multi-Layer Perceptron, the project has been upgraded to a **Convolutional Neural Network (CNN) powered by PyTorch**. 

The main objective of this architecture shift is to solve a classic machine learning challenge: bridging the performance gap between static dataset testing and dynamic, real-world user drawings on a web interface. By implementing a CNN alongside a rigorous preprocessing and data-augmentation pipeline, the system achieves excellent spatial invariance and high inference accuracy on live user input.

---

## 🧠 Network Architecture
The model transitions the image data from a raw grid into high-level spatial features using a compact, efficient CNN pipeline:

* **Input Layer:** Handles a $1 \times 28 \times 28$ grayscale image tensor (reshaped from the user's drawing canvas).
* **Convolutional Layer (`Conv2d`):** 16 filters with a $3 \times 3$ kernel size and padding of 1. This layer scans the canvas to extract local spatial features like edges, intersections, and curves, regardless of their position.
* **Activation:** ReLU (Rectified Linear Unit) to introduce non-linearity.
* **Max Pooling Layer (`MaxPool2d`):** A $2 \times 2$ window with a stride of 2 that downsamples the feature maps from $28 \times 28$ to $14 \times 14$, selecting only the most dominant spatial activations.
* **Fully Connected Layer (`Linear`):** Flattens the $16 \times 14 \times 14$ feature map into 128 dense nodes with a final ReLU activation.
* **Output Layer:** 10 nodes mapping directly to digits 0-9.

---

## ⚙️ Training & Live Data Augmentation
To ensure the network is resilient against diverse handwriting styles, stroke scales, and off-center drawings, the training pipeline incorporates a live **Data Augmentation** routine before batch ingestion:

1.  **Random Shifts:** Translates the images dynamically by $\pm2$ pixels vertically and horizontally using `scipy.ndimage.shift` to simulate variable placement on the canvas.
2.  **Random Zooms:** Applies a dynamic scaling factor between 0.9x and 1.1x using `scipy.ndimage.zoom` to simulate variations in writing pressure and pen stroke sizes.
3.  **Mini-Batch Optimization:** The network is optimized using the **Adam Optimizer** and **Cross-Entropy Loss**, processing data in mini-batches of 64 images for rapid convergence and superior generalization compared to full-batch gradient descent.

---

## 🎨 Real-Time Inference & Image Preprocessing
When a user draws a digit in the Streamlit UI, the application bridges the gap between the messy $450 \times 450$ canvas and the expected $28 \times 28$ model resolution. The custom preprocessing pipeline performs the following geometric operations:

1.  **Bounding Box Isolation:** Finds the precise outermost boundaries of the non-zero (drawn) pixels, stripping away dead margins.
2.  **Aspect-Ratio Scaling:** Scales the cropped drawing so its largest dimension is exactly 20 pixels, using bilinear/Lanczos resampling to match the anti-aliased look of the MNIST dataset.
3.  **Center-Mass Pasting:** Pastes the scaled digit precisely into the mathematical center of a brand new $28 \times 28$ black canvas.
4.  **Tensor Normalization:** Normalizes the pixel intensities to a `[0.0, 1.0]` range and reformats the array into a 4D tensor `(1, 1, 28, 28)` required for PyTorch CNN inference.

---

## 🚀 How to Run Locally

### 1. Clone the Repository & Install Dependencies
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install torch numpy scipy pillow streamlit streamlit-drawable-canvas
