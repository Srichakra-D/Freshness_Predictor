import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
from pathlib import Path

# Set the page configuration
st.set_page_config(
    page_title="Fruit Freshness Predictor",
    page_icon="🍎",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title of the app
st.title("🍎 Fruit Freshness Predictor")

# Description
st.write("""
This application predicts the type of fruit or vegetable and its freshness based on the uploaded image.
""")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_NAMES = (
    'apples',
    'banana',
    'bittergroud',
    'capsicum',
    'cucumber',
    'okra',
    'oranges',
    'potato',
    'tomato',
)

FRESHNESS_LABELS = ('Fresh', 'Spoiled')

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
    ),
])

# Define the ModelVGG16 class
class ModelVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.7
        
        self.base = models.vgg16(weights=None)
        
        # Freeze all layers except the last 15
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
                    
        self.base.classifier = nn.Sequential()  # Clear classifier
        self.base.fc = nn.Sequential()  # Remove fc layers
            
        # Custom blocks
        self.block1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),  # Adjust input size based on VGG16 output
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, len(CLASS_NAMES))
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, len(FRESHNESS_LABELS))
        )

    def forward(self, x):
        x = self.base.features(x)  # Use VGG16's convolutional layers
        x = torch.flatten(x, 1)    # Flatten the output
        x = self.block1(x)         # Pass through custom block1
        y1, y2 = self.block2(x), self.block3(x)  # Get predictions from block2 and block3
        return y1, y2

# Function to load the model with caching to prevent reloading on every run
@st.cache_resource
def load_model(model_path):
    model = ModelVGG16().to(device)
    try:
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        st.success(f"Model loaded successfully from `{model_path}`")
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()
    return model

# Load the model
model_path = Path(__file__).with_name("model.pth")
model_vgg16 = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert('RGB')
    return INFERENCE_TRANSFORM(image).unsqueeze(0)

# Function to run the prediction
def predict_freshness(image):
    image = preprocess_image(image)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model_vgg16(image)
        # Assuming the model has two outputs for fruit type and freshness
        fruit_pred = torch.argmax(outputs[0], dim=1).item()
        fresh_pred = torch.argmax(outputs[1], dim=1).item()
    
    # Map the predictions to labels
    fruit_label = CLASS_NAMES[fruit_pred]
    freshness_label = FRESHNESS_LABELS[fresh_pred]
    
    return fruit_label, freshness_label

# File uploader allows user to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Run prediction
        fruit_label, freshness_label = predict_freshness(image)
        
        # Display the results
        st.success(f"**Prediction:** {fruit_label}, {freshness_label}")
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Please upload an image to get started.")

# Footer
st.write("---")
