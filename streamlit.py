import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image

# Import the forward propagation math from your other file!
from neural_network import forward_prop

# 1. Force the app to use the full width of the screen
st.set_page_config(layout="wide")

st.title("🔢 Handwritten Digit Classifier - Shreeraj Chimanpure")
st.write("Draw a digit from 0-9 in the canvas below and see the neural network predict it!")


# --- LOAD THE MODEL WEIGHTS ---
# st.cache_resource ensures we only load the file once to keep the app fast
@st.cache_resource
def load_weights():
    try:
        data = np.load("mnist_weights.npz")
        return data['W1'], data['b1'], data['W2'], data['b2']
    except FileNotFoundError:
        return None, None, None, None


W1, b1, W2, b2 = load_weights()

if W1 is None:
    st.error("⚠️ Could not find 'mnist_weights.npz'. Make sure you run neural_network.py first!")
    st.stop()

# 2. Create Tabs
tab_app, tab_details = st.tabs(["🎮 Play Area", "🧠 Model Architecture"])

with tab_app:
    col_canvas, col_results = st.columns([1, 2])

    with col_canvas:
        st.subheader("Draw Here")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=50,
            stroke_color="#FFFFFF",  # White ink
            background_color="#000000",  # Black background (Matches MNIST format!)
            update_streamlit=True,
            height=450,
            width=450,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col_results:
        st.subheader("Prediction Results")

        with st.container(border=True):
            if canvas_result.image_data is not None:
                # The canvas returns a 3D numpy array (450, 450, 4) for RGBA.
                # We just need one color channel since it's black and white.
                img_array = canvas_result.image_data[:, :, 0]

                # Check if the user actually drew something (if the array isn't all zeros)
                if np.any(img_array > 0):
                    # --- IMAGE PROCESSING ---
                    if np.any(img_array > 0):
                        # --- NEW IMAGE PROCESSING ---
                        # 1. Find the bounding box of the drawing
                        coords = np.argwhere(img_array > 0)
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)

                        # 2. Crop the raw image to exactly what the user drew
                        cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]

                        # 3. Convert to PIL Image and resize, keeping the aspect ratio,
                        # so the largest dimension is 20 pixels (just like MNIST!)
                        img = Image.fromarray(cropped.astype('uint8'))
                        width, height = img.size
                        ratio = 20.0 / max(width, height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        # We use LANCZOS for high-quality downsampling
                        try:
                            resample_filter = Image.Resampling.LANCZOS
                        except AttributeError:
                            resample_filter = Image.LANCZOS  # Fallback for older Pillow versions

                        img_resized = img.resize((new_width, new_height), resample_filter)

                        # 4. Create a brand new 28x28 black canvas
                        final_img = Image.new('L', (28, 28), color=0)

                        # 5. Paste the resized digit right in the mathematical center
                        offset_x = (28 - new_width) // 2
                        offset_y = (28 - new_height) // 2
                        final_img.paste(img_resized, (offset_x, offset_y))

                        # 6. Convert back to array, flatten, and normalize
                        img_final_array = np.array(final_img)
                        X_input = img_final_array.reshape(784, 1) / 255.0

                        # --- MAKE PREDICTION ---
                        # Push the image through the math we wrote!
                        _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_input)

                    # --- MAKE PREDICTION ---
                    # Push the image through the math we wrote!
                    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X_input)

                    # A2 is an array of 10 probabilities.
                    # np.argmax gets the index of the highest probability (which is our digit!)
                    predicted_digit = np.argmax(A2)
                    confidence = np.max(A2)

                    # --- UI UPDATE ---
                    st.markdown("### Predicted Digit:")
                    st.markdown(
                        f"<h1 style='text-align: center; color: #FF4B4B; font-size: 80px;'>{predicted_digit}</h1>",
                        unsafe_allow_html=True)

                    st.write(f"Confidence: **{confidence * 100:.2f}%**")
                    st.progress(float(confidence))
                else:
                    st.info("Draw something on the canvas to see the prediction.")

with tab_details:
    st.subheader("Neural Network Details")
    st.write(
        "This is **not** a standard TensorFlow/PyTorch model. This is a custom Artificial Neural Network built entirely from scratch using only raw linear algebra and calculus (NumPy).")

    st.markdown("""
    **Architecture Overview:**
    * **Input Layer:** 784 nodes (28x28 pixel flattened image)
    * **Hidden Layer:** 128 nodes using a ReLU (Rectified Linear Unit) activation function.
    * **Output Layer:** 10 nodes (Digits 0-9) using a Softmax activation function to convert raw scores into probabilities.
    * **Training:** Full Batch Gradient Descent over 500 iterations.
    """)
