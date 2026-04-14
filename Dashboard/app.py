import streamlit as st
import json
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

st.set_page_config(
    page_title="SpaceLandMapper",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent.parent

baseline_metrics_path = BASE_DIR / "Outputs" / "baseline" / "metrics.json"
cnn_metrics_path = BASE_DIR / "Outputs" / "cnn" / "metrics.json"

baseline_cm_path = BASE_DIR / "Outputs" / "baseline" / "confusion_matrix.png"
cnn_cm_path = BASE_DIR / "Outputs" / "cnn" / "confusion_matrix.png"
cnn_model_path = BASE_DIR / "Outputs" / "cnn" / "cnn_model.pth"
cnn_predictions_path = BASE_DIR / "Outputs" / "cnn" / "predictions.csv"
val_csv_path = BASE_DIR / "Data" / "val.csv"

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


def load_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


baseline_metrics = load_json(baseline_metrics_path)
cnn_metrics = load_json(cnn_metrics_path)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
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


@st.cache_data
def load_prediction_examples():
    if not cnn_predictions_path.exists() or not val_csv_path.exists():
        return pd.DataFrame()

    preds = pd.read_csv(cnn_predictions_path)
    val_df = pd.read_csv(val_csv_path)

    if len(preds) != len(val_df):
        return pd.DataFrame()

    merged = val_df.copy()
    merged["true_label"] = preds["true_label"]
    merged["predicted_label"] = preds["predicted_label"]
    merged["is_correct"] = merged["true_label"] == merged["predicted_label"]
    return merged


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


baseline_acc = baseline_metrics.get("accuracy", 0.0)
cnn_acc = cnn_metrics.get("accuracy", 0.0)
baseline_f1 = baseline_metrics.get("macro_f1", 0.0)
cnn_f1 = cnn_metrics.get("macro_f1", 0.0)

acc_gain = cnn_acc - baseline_acc
f1_gain = cnn_f1 - baseline_f1

examples_df = load_prediction_examples()

st.markdown("""
<style>
:root {
    --bg-1: #06111f;
    --bg-2: #0b1830;
    --bg-3: #101d45;
    --card: rgba(13, 22, 42, 0.68);
    --card-2: rgba(19, 31, 58, 0.78);
    --line: rgba(255,255,255,0.14);
    --text: #f7fbff;
    --muted: #d7e4ff;
    --cyan: #67e8f9;
    --blue: #60a5fa;
    --purple: #a78bfa;
    --pink: #f472b6;
    --gold: #facc15;
    --green: #34d399;
    --red: #fb7185;
}

html, body, [class*="css"] {
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at 8% 12%, rgba(103,232,249,0.30), transparent 18%),
        radial-gradient(circle at 90% 10%, rgba(167,139,250,0.30), transparent 18%),
        radial-gradient(circle at 78% 88%, rgba(244,114,182,0.26), transparent 18%),
        radial-gradient(circle at 20% 82%, rgba(250,204,21,0.20), transparent 16%),
        linear-gradient(135deg, #07111f 0%, #0c1732 25%, #13245b 52%, #252f72 72%, #4b2c82 88%, #6f2ca0 100%);
    background-attachment: fixed;
}

.block-container {
    max-width: 1280px;
    padding-top: 1.25rem;
    padding-bottom: 2.8rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

@keyframes floatGlow {
    0% { box-shadow: 0 18px 40px rgba(0,0,0,0.20), 0 0 0 rgba(103,232,249,0.00); }
    50% { box-shadow: 0 22px 48px rgba(0,0,0,0.24), 0 0 36px rgba(103,232,249,0.14); }
    100% { box-shadow: 0 18px 40px rgba(0,0,0,0.20), 0 0 0 rgba(103,232,249,0.00); }
}

@keyframes sweep {
    0% { transform: translateX(-130%) skewX(-18deg); }
    100% { transform: translateX(230%) skewX(-18deg); }
}

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 32px;
    padding: 38px 40px 34px 40px;
    margin-bottom: 24px;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.05)),
        linear-gradient(135deg, rgba(103,232,249,0.16), rgba(96,165,250,0.10), rgba(167,139,250,0.16), rgba(244,114,182,0.14));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.16);
    animation: floatGlow 4.5s ease-in-out infinite;
}

.hero::before {
    content: "";
    position: absolute;
    top: 0;
    left: -25%;
    width: 18%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    animation: sweep 6.5s linear infinite;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 16px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: white;
    background: linear-gradient(90deg, rgba(103,232,249,0.26), rgba(167,139,250,0.26), rgba(244,114,182,0.22));
    border: 1px solid rgba(255,255,255,0.16);
}

.hero-title {
    font-size: 62px;
    font-weight: 900;
    line-height: 1.0;
    margin-bottom: 14px;
    background: linear-gradient(90deg, #ffffff, #dbeafe, #e9d5ff, #fde68a, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 18px;
    line-height: 1.78;
    color: rgba(255,255,255,0.93);
    max-width: 930px;
}

.section-title {
    font-size: 30px;
    font-weight: 900;
    margin: 6px 0 16px 0;
    color: white;
    letter-spacing: 0.02em;
    text-shadow: 0 2px 10px rgba(0,0,0,0.14);
}

.card,
.card-soft,
.metric-card,
.model-card,
.prediction-card,
.class-card,
.example-card,
.badge-card {
    backdrop-filter: blur(14px);
    border: 1px solid var(--line);
    box-shadow: 0 18px 36px rgba(0,0,0,0.16);
}

.card {
    background: linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
    border-radius: 26px;
    padding: 22px;
}

.card-soft {
    background: linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0.05));
    border-radius: 22px;
    padding: 22px;
}

.metric-grid-note {
    color: rgba(255,255,255,0.90);
    line-height: 1.7;
}

.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.18), rgba(255,255,255,0.07));
    border-radius: 24px;
    padding: 18px 16px;
    text-align: center;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}

.metric-card:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 22px 40px rgba(103,232,249,0.16);
}

.metric-label {
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.86);
    margin-bottom: 10px;
}

.metric-value {
    font-size: 36px;
    font-weight: 900;
    color: white;
    text-shadow: 0 4px 18px rgba(255,255,255,0.14);
}

.model-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
    border-radius: 28px;
    padding: 24px;
}

.model-title {
    font-size: 24px;
    font-weight: 900;
    margin-bottom: 8px;
    color: white;
}

.model-subtitle {
    color: rgba(255,255,255,0.88);
    margin-bottom: 18px;
    line-height: 1.7;
}

.kpi-chip {
    display: inline-block;
    padding: 10px 14px;
    margin: 4px 8px 4px 0;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(103,232,249,0.18), rgba(167,139,250,0.18));
    border: 1px solid rgba(255,255,255,0.14);
    font-weight: 800;
}

.highlight {
    color: #fff3c4;
    font-weight: 900;
}

.subtle {
    color: rgba(255,255,255,0.92);
    line-height: 1.72;
}

.badge-card {
    background: linear-gradient(135deg, rgba(103,232,249,0.22), rgba(167,139,250,0.20), rgba(244,114,182,0.20));
    border-radius: 24px;
    padding: 18px;
    text-align: center;
}

.badge-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.88);
    margin-bottom: 8px;
    font-weight: 800;
}

.badge-value {
    font-size: 30px;
    font-weight: 900;
    color: white;
}

.prediction-card {
    background: linear-gradient(145deg, rgba(52,211,153,0.16), rgba(255,255,255,0.08));
    border-radius: 28px;
    padding: 24px;
}

.class-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
    border-radius: 20px;
    padding: 16px;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}

.class-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 34px rgba(167,139,250,0.18);
}

.class-number {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 800;
    color: rgba(255,255,255,0.84);
    margin-bottom: 6px;
}

.class-name {
    font-size: 16px;
    font-weight: 900;
    line-height: 1.35;
    color: white;
    word-break: break-word;
}

.example-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.14), rgba(255,255,255,0.05));
    border-radius: 20px;
    padding: 14px;
}

.spacer {
    height: 10px;
}

[data-testid="stTabs"] {
    margin-top: 8px;
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 10px;
    background: linear-gradient(90deg, rgba(255,255,255,0.12), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.14);
    padding: 10px;
    border-radius: 24px;
    backdrop-filter: blur(14px);
    box-shadow: 0 14px 28px rgba(0,0,0,0.12);
}

[data-testid="stTabs"] [role="tab"] {
    height: 54px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.04));
    color: white;
    font-weight: 900;
    padding: 0 20px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.2s ease;
}

[data-testid="stTabs"] [role="tab"]:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(255,255,255,0.18);
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(103,232,249,0.28), rgba(96,165,250,0.24), rgba(167,139,250,0.26), rgba(244,114,182,0.24));
    color: white;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 12px 24px rgba(103,232,249,0.12);
}

.stMetric {
    background: transparent !important;
}

[data-testid="stImage"] img {
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 16px 34px rgba(0,0,0,0.14);
}

[data-testid="stFileUploader"] section {
    background: linear-gradient(145deg, rgba(255,255,255,0.16), rgba(255,255,255,0.06));
    border-radius: 20px;
    border: 1px dashed rgba(255,255,255,0.24);
    box-shadow: 0 12px 28px rgba(0,0,0,0.12);
}

.stAlert {
    border-radius: 16px;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="badge-row">
        <div class="hero-badge">Earth Observation</div>
        <div class="hero-badge">AI Solution</div>
        <div class="hero-badge">Land Classification</div>
        <div class="hero-badge">Interactive Prototype</div>
    </div>
    <div class="hero-title">🛰️ SpaceLandMapper</div>
    <div class="hero-subtitle">
        SpaceLandMapper is an AI solution for land-use classification using EuroSAT imagery derived from Sentinel-2 satellite data.
        The prototype supports more efficient land mapping for urban planning, environmental conservation, and resource management,
        while comparing a baseline model and a CNN to identify the stronger final approach.
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Model Comparison",
    "Confusion Matrices",
    "Live Demo",
    "Prediction Examples",
    "Classes & Conclusion",
])

with tab1:
    st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)

    left, right = st.columns([1.7, 1])

    with left:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0; color:white;">About the Project</h3>
            <p class="subtle">
            SpaceLandMapper addresses the challenge of land-use mapping by applying AI to satellite imagery.
            The project focuses on improving classification accuracy using EuroSAT data and demonstrating the final solution through an interactive dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="card">
            <h3 style="margin-top:0; color:white;">Why This Matters</h3>
            <p class="subtle">• Supports urban planning and land analysis</p>
            <p class="subtle">• Helps improve efficiency over manual methods</p>
            <p class="subtle">• Demonstrates AI for Earth observation</p>
            <p class="subtle">• Provides a practical land classification prototype</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">EuroSAT</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Classes</div>
            <div class="metric-value">10</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Input Size</div>
            <div class="metric-value">64×64</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-soft">
        <h3 style="margin-top:0; color:white;">How It Works</h3>
        <p class="metric-grid-note">
        EuroSAT image patches are prepared and used to train two models: a baseline classifier and a CNN.
        Their results are compared using accuracy, Macro F1, and confusion matrices, and the stronger model is presented in the dashboard for live prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown('<div class="model-title">Baseline Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-subtitle">Baseline model used as the initial benchmark for land classification performance.</div>', unsafe_allow_html=True)
        if baseline_metrics:
            st.write(f"**Model:** {baseline_metrics.get('model', 'N/A')}")
            st.write(f"**Accuracy:** {baseline_metrics.get('accuracy', 0):.4f}")
            st.write(f"**Macro F1:** {baseline_metrics.get('macro_f1', 0):.4f}")
        else:
            st.warning("Baseline metrics not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="model-card">', unsafe_allow_html=True)
        st.markdown('<div class="model-title">CNN Model</div>', unsafe_allow_html=True)
        st.markdown('<div class="model-subtitle">CNN model used as the final AI solution after achieving stronger classification results.</div>', unsafe_allow_html=True)
        if cnn_metrics:
            st.write(f"**Model:** {cnn_metrics.get('model', 'N/A')}")
            st.write(f"**Accuracy:** {cnn_metrics.get('accuracy', 0):.4f}")
            st.write(f"**Macro F1:** {cnn_metrics.get('macro_f1', 0):.4f}")
        else:
            st.warning("CNN metrics not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        st.metric("Accuracy Gain (CNN - Baseline)", f"{acc_gain:.4f}")
    with b2:
        st.metric("Macro F1 Gain (CNN - Baseline)", f"{f1_gain:.4f}")

    st.markdown(
        f'<div class="kpi-chip">Baseline Accuracy: {baseline_acc:.4f}</div>'
        f'<div class="kpi-chip">CNN Accuracy: {cnn_acc:.4f}</div>'
        f'<div class="kpi-chip">Baseline Macro F1: {baseline_f1:.4f}</div>'
        f'<div class="kpi-chip">CNN Macro F1: {cnn_f1:.4f}</div>',
        unsafe_allow_html=True
    )

    compare_df = pd.DataFrame({
        "Model": ["Baseline", "CNN"],
        "Accuracy": [baseline_acc, cnn_acc],
        "Macro F1": [baseline_f1, cnn_f1],
    }).set_index("Model")

    st.bar_chart(compare_df)

    st.markdown(f"""
    <div class="card-soft">
        <p class="subtle">
        The CNN achieved stronger classification performance than the baseline model, so it was selected as the final model for the SpaceLandMapper solution.
        </p>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-soft">
        <h3 style="margin-top:0; color:white;">What the Confusion Matrices Show</h3>
        <p class="subtle">
        A confusion matrix shows how often each land class is predicted correctly and where the model confuses one class with another.
        Comparing both matrices highlights the improvement achieved by the CNN.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    x1, x2, x3, x4 = st.columns(4)
    with x1:
        st.markdown(f"""
        <div class="badge-card">
            <div class="badge-title">Baseline Accuracy</div>
            <div class="badge-value">{baseline_acc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with x2:
        st.markdown(f"""
        <div class="badge-card">
            <div class="badge-title">Baseline Macro F1</div>
            <div class="badge-value">{baseline_f1:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with x3:
        st.markdown(f"""
        <div class="badge-card">
            <div class="badge-title">CNN Accuracy</div>
            <div class="badge-value">{cnn_acc:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with x4:
        st.markdown(f"""
        <div class="badge-card">
            <div class="badge-title">CNN Macro F1</div>
            <div class="badge-value">{cnn_f1:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    img1, img2 = st.columns(2)

    with img1:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">Baseline Confusion Matrix</div>
            <div class="model-subtitle">
                The baseline model shows weaker class separation and more confusion between similar land categories.
            </div>
        """, unsafe_allow_html=True)
        if baseline_cm_path.exists():
            st.image(str(baseline_cm_path), use_container_width=True)
        else:
            st.warning("Baseline confusion matrix not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    with img2:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">CNN Confusion Matrix</div>
            <div class="model-subtitle">
                The CNN shows stronger class separation and more accurate land classification.
            </div>
        """, unsafe_allow_html=True)
        if cnn_cm_path.exists():
            st.image(str(cnn_cm_path), use_container_width=True)
        else:
            st.warning("CNN confusion matrix not found.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    i_left, i_right = st.columns([3, 1])
    with i_left:
        st.markdown(f"""
        <div class="card-soft">
            <h3 style="margin-top:0; color:white;">Interpretation</h3>
            <p class="subtle">
            The CNN confusion matrix is cleaner than the baseline matrix, which supports the improvement shown in the evaluation metrics.
            This means the final model is making more correct predictions and reducing confusion across land classes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with i_right:
        st.markdown("""
        <div class="badge-card">
            <div class="badge-title">Selected Model</div>
            <div class="badge-value">CNN</div>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-title">Live Demo</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card-soft">
        <p class="subtle">
        Upload a land image to test the final CNN-based land classification solution in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a JPG or PNG land image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        d1, d2 = st.columns([1, 1])

        with d1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with d2:
            if cnn_model_path.exists():
                with st.spinner("Analyzing satellite patch..."):
                    pred_label, probs = predict_image(image)

                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.subheader("Prediction Result")
                st.success(f"Predicted Class: {pred_label}")

                top3_idx = np.argsort(probs)[::-1][:3]
                top3_df = pd.DataFrame({
                    "Class": [CLASS_NAMES[i] for i in top3_idx],
                    "Confidence": [float(probs[i]) for i in top3_idx],
                })
                st.write("Top 3 Predictions")
                st.dataframe(top3_df, use_container_width=True, hide_index=True)

                confidence_dict = {
                    CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
                }
                st.write("Confidence Scores")
                st.bar_chart(confidence_dict)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("CNN model file not found.")

with tab5:
    st.markdown('<div class="section-title">Prediction Examples</div>', unsafe_allow_html=True)

    if examples_df.empty:
        st.warning("Prediction examples could not be loaded.")
    else:
        correct_examples = examples_df[examples_df["is_correct"]].head(3)
        wrong_examples = examples_df[~examples_df["is_correct"]].head(3)

        st.markdown("""
        <div class="card-soft">
            <p class="subtle">
            These examples show where the final CNN model performs well and where some classification errors still occur.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

        st.subheader("Correct Predictions")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(correct_examples.iterrows()):
            with cols[idx]:
                st.markdown('<div class="example-card">', unsafe_allow_html=True)
                if Path(row["image_path"]).exists():
                    st.image(row["image_path"], use_container_width=True)
                st.success(f"True: {row['true_label']}")
                st.info(f"Predicted: {row['predicted_label']}")
                st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("Misclassified Examples")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(wrong_examples.iterrows()):
            with cols[idx]:
                st.markdown('<div class="example-card">', unsafe_allow_html=True)
                if Path(row["image_path"]).exists():
                    st.image(row["image_path"], use_container_width=True)
                st.error(f"True: {row['true_label']}")
                st.warning(f"Predicted: {row['predicted_label']}")
                st.markdown('</div>', unsafe_allow_html=True)

with tab6:
    st.markdown('<div class="section-title">Land Classes</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, class_name in enumerate(CLASS_NAMES):
        with cols[i % 5]:
            st.markdown(
                f"""
                <div class="class-card">
                    <div class="class-number">Class {i + 1}</div>
                    <div class="class-name">{class_name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Limitations & Future Work</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-soft">
        <p class="subtle">
        <b>Current limitations:</b> the prototype is based on EuroSAT benchmark imagery and does not yet implement full real-time land-use change monitoring.
        </p>
        <p class="subtle">
        <b>Future improvements:</b> train for more epochs, apply data augmentation, test broader Sentinel-2 imagery, and extend the system toward wider land monitoring tasks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Final Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card-soft">
        <p class="subtle">
        This prototype shows how AI can support land-use mapping from Earth observation imagery.
        The CNN achieved stronger performance than the baseline and was selected as the final model for the SpaceLandMapper solution.
        </p>
    </div>
    """, unsafe_allow_html=True)