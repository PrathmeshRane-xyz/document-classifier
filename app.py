
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Type Classification")
st.write(
    "Upload a document image or capture one using your camera. "
    "The model will predict its category."
)

# -----------------------------
# Model configuration
# -----------------------------
MODEL_PATH = "document_classifier_final.pth"
DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "Email",
    "Form",
    "Letter",
    "Memo",
    "Note",
    "Report"
]

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint.get("class_names", CLASS_NAMES)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(class_names)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Prediction function
# -----------------------------
def predict_document(image, model, class_names):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = probabilities.argmax().item()

    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    return predicted_class, confidence, probabilities


# -----------------------------
# Load model safely
# -----------------------------
try:
    model, class_names = load_model()
    st.success("Model loaded successfully!")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# -----------------------------
# Input options
# -----------------------------
st.subheader("Choose your input")

input_method = st.radio(
    "How would you like to provide a document?",
    ["Upload an image", "Capture using camera"],
    horizontal=True
)

image = None

if input_method == "Upload an image":
    uploaded_file = st.file_uploader(
        "Choose a document image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

else:
    camera_image = st.camera_input("Take a picture of your document")

    if camera_image is not None:
        image = Image.open(camera_image)


# -----------------------------
# Display and classify
# -----------------------------
if image is not None:
    st.subheader("Document preview")
    st.image(image, caption="Input document", use_container_width=True)

    if st.button("Classify Document", type="primary"):
        with st.spinner("Classifying document..."):
            predicted_class, confidence, probabilities = predict_document(
                image, model, class_names
            )

        st.subheader("Prediction result")
        st.success(f"Predicted category: {predicted_class}")
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.subheader("Class probabilities")

        for index, class_name in enumerate(class_names):
            probability = probabilities[index].item()
            st.write(f"**{class_name}:** {probability * 100:.2f}%")
            st.progress(probability)

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )
