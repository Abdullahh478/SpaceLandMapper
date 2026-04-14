import streamlit as st
import json
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

st.set_page_config(page_title="SpaceLandMapper", page_icon="🛰️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent

baseline_metrics_path = BASE_DIR / "Outputs" / "baseline" / "metrics.json"
cnn_metrics_path = BASE_DIR / "Outputs" / "cnn" / "metrics.json"

baseline_cm_path = BASE_DIR / "Outputs" / "baseline" / "confusion_matrix.png"
cnn_cm_path = BASE_DIR / "Outputs" / "cnn" / "confusion_matrix.png"
cnn_model_path = BASE_DIR / "Outputs" / "cnn" / "cnn_model.pth"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

IMAGE_SIZE = (64, 64)

baseline_metrics = {}
cnn_metrics = {}

if baseline_metrics_path.exists():
    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

if cnn_metrics_path.exists():
    with open(cnn_metrics_path, "r", encoding="utf-8") as f:
        cnn_metrics = json.load(f)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


@st.cache_resource
def load_model():
    model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return CLASS_NAMES[pred_idx], probs.numpy()


st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    font-size: 52px;
    font-weight: 800;
    color: white;
    margin-bottom: 6px;
}
.subtitle {
    font-size: 18px;
    color: #b8c0d0;
    margin-bottom: 30px;
}
.section-title {
    font-size: 30px;
    font-weight: 700;
    color: white;
    margin-top: 25px;
    margin-bottom: 18px;
}
.info-card {
    background: linear-gradient(145deg, #101827, #0b1220);
    border: 1px solid #1f2a44;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    min-height: 180px;
}
.metric-card {
    background: linear-gradient(145deg, #13203a, #0c1627);
    border: 1px solid #223253;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.20);
}
.model-card {
    background: linear-gradient(145deg, #101827, #0b1220);
    border: 1px solid #1f2a44;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
}
.highlight {
    color: #7dd3fc;
    font-weight: 700;
}
.small-note {
    color: #b8c0d0;
    font-size: 15px;
}
.prediction-box {
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
}
.class-card {
    background: linear-gradient(145deg, #13203a, #0c1627);
    border: 1px solid #223253;
    border-radius: 16px;
    padding: 14px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    margin-bottom: 12px;
}
.class-number {
    font-size: 14px;
    color: #b8c0d0;
    margin-bottom: 6px;
}
.class-name {
    font-size: 17px;
    font-weight: 700;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛰️ SpaceLandMapper Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-based land classification using EuroSAT satellite imagery, comparing a baseline Logistic Regression model against a Convolutional Neural Network (CNN).</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="section-title">Project Summary</div>', unsafe_allow_html=True)

overview_col1, overview_col2 = st.columns([2, 1])

with overview_col1:
    st.markdown("""
    <div class="info-card">
        <h3>About the Project</h3>
        <p class="small-note">
        SpaceLandMapper is a land classification project built using EuroSAT imagery derived from Sentinel-2 satellite data.
        The goal is to classify land-use categories and compare the performance of a traditional baseline model with a CNN-based deep learning approach.
        </p>
        <p class="small-note">
        The project supports applications in <span class="highlight">environmental monitoring</span>,
        <span class="highlight">urban planning</span>, <span class="highlight">agriculture</span>,
        and <span class="highlight">resource management</span>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with overview_col2:
    st.markdown("""
    <div class="info-card">
        <h3>Team Focus</h3>
        <p class="small-note"><b>Dataset:</b> EuroSAT</p>
        <p class="small-note"><b>Classes:</b> 10</p>
        <p class="small-note"><b>Input Size:</b> 64 × 64</p>
        <p class="small-note"><b>Models:</b> Baseline + CNN</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">Key Project Metrics</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Dataset", "EuroSAT")
with m2:
    st.metric("Classes", "10")
with m3:
    st.metric("Image Size", "64×64")

st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.subheader("Baseline Model")
    if baseline_metrics:
        st.write(f"**Model:** {baseline_metrics.get('model', 'N/A')}")
        st.write(f"**Accuracy:** {baseline_metrics.get('accuracy', 0):.4f}")
        st.write(f"**Macro F1:** {baseline_metrics.get('macro_f1', 0):.4f}")
    else:
        st.warning("Baseline metrics not found.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="model-card">', unsafe_allow_html=True)
    st.subheader("CNN Model")
    if cnn_metrics:
        st.write(f"**Model:** {cnn_metrics.get('model', 'N/A')}")
        st.write(f"**Accuracy:** {cnn_metrics.get('accuracy', 0):.4f}")
        st.write(f"**Macro F1:** {cnn_metrics.get('macro_f1', 0):.4f}")
    else:
        st.warning("CNN metrics not found.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)

img_col1, img_col2 = st.columns(2)

with img_col1:
    st.subheader("Baseline Confusion Matrix")
    if baseline_cm_path.exists():
        st.image(str(baseline_cm_path), use_container_width=True)
    else:
        st.warning("Baseline confusion matrix not found.")

with img_col2:
    st.subheader("CNN Confusion Matrix")
    if cnn_cm_path.exists():
        st.image(str(cnn_cm_path), use_container_width=True)
    else:
        st.warning("CNN confusion matrix not found.")

st.markdown('<div class="section-title">Live CNN Prediction Demo</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a land image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    demo_col1, demo_col2 = st.columns([1, 1])

    with demo_col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with demo_col2:
        if cnn_model_path.exists():
            pred_label, probs = predict_image(image)
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader("Prediction Result")
            st.success(f"Predicted Class: {pred_label}")

            confidence_dict = {
                CLASS_NAMES[i]: float(probs[i])
                for i in range(len(CLASS_NAMES))
            }
            st.write("Confidence Scores")
            st.bar_chart(confidence_dict)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("CNN model file not found.")

st.markdown('<div class="section-title">Land Classes</div>', unsafe_allow_html=True)

class_cols = st.columns(5)
for i, class_name in enumerate(CLASS_NAMES):
    with class_cols[i % 5]:
        st.markdown(
            f"""
            <div class="class-card">
                <div class="class-number">Class {i + 1}</div>
                <div class="class-name">{class_name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown('<div class="section-title">Performance Insight</div>', unsafe_allow_html=True)

baseline_acc = baseline_metrics.get("accuracy", 0)
cnn_acc = cnn_metrics.get("accuracy", 0)
baseline_f1 = baseline_metrics.get("macro_f1", 0)
cnn_f1 = cnn_metrics.get("macro_f1", 0)

acc_gain = cnn_acc - baseline_acc
f1_gain = cnn_f1 - baseline_f1

st.markdown(f"""
<div class="info-card">
    <p class="small-note">
    The CNN achieved a clear improvement over the baseline model.
    Accuracy improved by <span class="highlight">{acc_gain:.4f}</span> and Macro F1 improved by
    <span class="highlight">{f1_gain:.4f}</span>, showing that the CNN was much more effective
    for the EuroSAT land classification task.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Final Conclusion</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <p class="small-note">
    This prototype demonstrates that deep learning can significantly improve land classification performance over a simpler baseline approach.
    The final SpaceLandMapper dashboard combines evaluation results, visual analysis, and live image prediction in one interface,
    making the solution suitable for an academic demonstration of AI-assisted Earth observation.
    </p>
</div>
""", unsafe_allow_html=True)