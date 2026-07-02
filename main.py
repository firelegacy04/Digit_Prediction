import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

# Import the PyTorch model class from your updated file
from neural_network import SmallCNN

# 1. Force the app to use the full width of the screen
st.set_page_config(layout="wide")

st.title("🔢 Handwritten Digit Classifier (CNN Edition) - Shreeraj C.")
st.write("Draw a digit from 0-9 in the canvas below and see the neural network predict it!")


# --- LOAD THE PYTORCH MODEL ---
# st.cache_resource ensures we only load the model once to keep the app fast
def load_model():
    try:
        model = SmallCNN()
        # map_location='cpu' ensures it runs on any machine, even without a GPU
        model.load_state_dict(torch.load("cnn_mnist.pth", map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode! Crucial for inference.
        return model
    except FileNotFoundError:
        return None


model = load_model()

if model is None:
    st.error("⚠️ Could not find 'cnn_mnist.pth'. Make sure you run neural_network.py first!")
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
            background_color="#000000",  # Black background (Matches MNIST format)
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
                # We just need one color channel since it's black and white.
                img_array = canvas_result.image_data[:, :, 0]

                raw_preview = Image.fromarray(img_array.astype("uint8"))
                raw_preview = raw_preview.filter(ImageFilter.GaussianBlur(radius=1))



                # Check if the user actually drew something
                if np.any(img_array > 0):
                    # --- IMAGE PROCESSING ---
                    # 1. Find the bounding box of the drawing
                    coords = np.argwhere(img_array > 0)
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)

                    # 2. Crop the raw image to exactly what the user drew
                    cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]

                    # 3. Convert to PIL Image and resize, keeping the aspect ratio
                    img = Image.fromarray(cropped.astype('uint8'))
                    width, height = img.size
                    ratio = 20.0 / max(width, height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)

                    try:
                        resample_filter = Image.Resampling.BILINEAR
                    except AttributeError:
                        resample_filter = Image.LANCZOS

                    img_resized = img.resize((new_width, new_height), resample_filter)

                    # 4. Create a brand new 28x28 black canvas
                    final_img = Image.new('L', (28, 28), color=0)

                    # 5. Paste the resized digit right in the mathematical center
                    offset_x = (28 - new_width) // 2
                    offset_y = (28 - new_height) // 2
                    final_img.paste(img_resized, (offset_x, offset_y))

                    # 6. Convert back to array and normalize
                    img_final_array = np.asarray(final_img, dtype=np.float32) / 255.0

                    # FORMAT FOR PYTORCH: (Batch_Size, Channels, Height, Width) -> (1, 1, 28, 28)
                    X_input = torch.tensor(img_final_array).view(1, 1, 28, 28)

                    # --- MAKE PREDICTION ---
                    with torch.no_grad():  # Tells PyTorch not to calculate gradients (saves memory)
                        raw_output = model(X_input)  # These are raw 'logits'

                        # Apply Softmax to convert raw logits into percentages (0 to 1)
                        probabilities = F.softmax(raw_output, dim=1).numpy()[0]

                    # Get the predicted digit and confidence
                    predicted_digit = np.argmax(probabilities)
                    confidence = probabilities[predicted_digit]

                    # --- UI UPDATE ---
                    st.markdown("### Predicted Digit:")
                    st.markdown(
                        f"<h1 style='text-align: center; color: #FF4B4B; font-size: 80px;'>{predicted_digit}</h1>",
                        unsafe_allow_html=True)

                    st.write(f"Confidence: **{confidence * 100:.2f}%**")

                    # Show exactly what the neural network receives
                    st.markdown("### 🖼️ Image Sent to the Network")
                    preview = final_img.resize((280, 280), Image.Resampling.NEAREST)
                    st.image(
                        preview,
                        caption="28×28 preprocessed image (10x enlarged)",
                        use_container_width=False
                    )



                    st.markdown("### ✏️ Raw Canvas")
                    st.image(raw_preview, width=250)

                    # Pass the probabilities to the bar chart
                    st.bar_chart(probabilities)
                    st.progress(float(confidence))
                else:
                    st.info("Draw something on the canvas to see the prediction.")

with tab_details:
    st.subheader("Neural Network Details")
    st.write(
        "This app is powered by a custom **Convolutional Neural Network (CNN)** built in PyTorch.")

    st.markdown("""
    **Architecture Overview:**
    * **Input:** 28x28 grayscale image (processed from canvas bounding box)
    * **Convolutional Layer:** 16 filters using a 3x3 kernel, followed by a ReLU activation. This extracts local spatial features like edges and curves.
    * **Max Pooling Layer:** Reduces the spatial dimensions from 28x28 down to 14x14, keeping only the strongest activations.
    * **Fully Connected (Dense) Layer 1:** 128 nodes using a ReLU activation.
    * **Output Layer:** 10 nodes (Digits 0-9).
    * **Training:** Mini-batch Gradient Descent (Adam Optimizer) with live Data Augmentation (shifts and zooms) to simulate human drawing variations.
    """)
