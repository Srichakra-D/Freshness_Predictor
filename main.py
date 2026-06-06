from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from inference import get_device, load_model, predict


st.set_page_config(
    page_title="Fruit Freshness Predictor",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title("🍎 Fruit Freshness Predictor")
st.write(
    "This application predicts the type of fruit or vegetable and its freshness "
    "based on the uploaded image."
)


@st.cache_resource
def load_cached_model(model_path: Path):
    return load_model(model_path, get_device())


model_path = Path(__file__).with_name("model.pth")
try:
    model_vgg16 = load_cached_model(model_path)
except (FileNotFoundError, RuntimeError, ValueError) as error:
    st.error(f"Unable to load the prediction model: {error}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as uploaded_image:
            image = uploaded_image.copy()

        st.image(image, caption="Uploaded image", use_container_width=True)
        result = predict(model_vgg16, image)

        st.success(f"**Prediction:** {result.fruit}, {result.freshness}")
        fruit_column, freshness_column = st.columns(2)
        fruit_column.metric("Produce confidence", f"{result.fruit_confidence:.1%}")
        freshness_column.metric(
            "Freshness confidence", f"{result.freshness_confidence:.1%}"
        )
        if result.is_low_confidence:
            st.warning(
                "This prediction has low confidence. Try a clear, well-lit image "
                "containing one supported fruit or vegetable."
            )
    except (UnidentifiedImageError, OSError, ValueError, TypeError) as error:
        st.error(f"Unable to process this image: {error}")
else:
    st.info("Please upload an image to get started.")

st.write("---")
