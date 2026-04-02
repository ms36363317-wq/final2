import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eye Disease AI Diagnosis",
    page_icon="👁️",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-badge {
        display: inline-block;
        background: #1a73e8;
        color: white;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .confidence-bar-label {
        font-size: 0.9rem;
        color: #333;
        margin-bottom: 4px;
    }
    .report-box {
        background: #f0f4ff;
        border-left: 4px solid #1a73e8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        font-size: 0.97rem;
        line-height: 1.8;
        color: #222;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #555;
        margin-top: 1.5rem;
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a73e8;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Diabetic Retinopathy',
    'Disc Edema',
    'Healthy',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa',
]

IMG_SIZE = (300, 300)
LAST_CONV_LAYER = "top_conv"

# ─── Load Models ─────────────────────────────────────────────────────────────
from tensorflow.keras.models import load_model

def load_vision_model():
    model = load_model("model.h5", compile=False)
    return model


def load_llm():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, llm

# ─── Prediction ──────────────────────────────────────────────────────────────
def predict(img_pil, model):
    img = img_pil.resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    return CLASS_NAMES[idx], confidence, preds[0], arr

# ─── Grad-CAM++ ──────────────────────────────────────────────────────────────
def make_gradcam_plusplus(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            conv_outputs, predictions = grad_model(img_array, training=False)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape1.gradient(loss, conv_outputs)
    second_grads = tape2.gradient(grads, conv_outputs)

    grads_np = grads.numpy()[0]
    second_grads_np = second_grads.numpy()[0]
    conv_np = conv_outputs.numpy()[0]

    # Grad-CAM++ weights
    denom = 2 * grads_np**2 + conv_np * second_grads_np + 1e-8
    alpha = second_grads_np / denom
    weights = np.sum(alpha * np.maximum(grads_np, 0), axis=(0, 1))

    cam = np.sum(weights * conv_np, axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Resize
    cam_resized = cv2.resize(cam, IMG_SIZE)
    heatmap = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    original_np = np.array(img_array[0])
    original_np = np.clip(original_np, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)

    return original_np, heatmap_color, overlay

# ─── LLM Report ──────────────────────────────────────────────────────────────
def generate_report(disease, confidence, tokenizer, llm):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = f"""You are an ophthalmology AI assistant.

Write exactly 5 short medical lines about this eye scan prediction:

Prediction: {disease}
Confidence: {confidence:.0%}

Structure:
1. Prediction statement.
2. Short definition of the condition.
3. Key symptoms the patient may experience.
4. Severity / urgency level.
5. Recommended next step.

Only write the 5 lines. No titles. No repetition."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    report = decoded[len(prompt):].strip()
    return report

# ─── UI ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">👁️ Eye Disease AI Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a retinal fundus image for AI-powered disease classification and medical report</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    use_gradcam = st.toggle("Show Grad-CAM++ heatmap", value=True)
    use_llm = st.toggle("Generate AI medical report (Phi-3)", value=True)
    st.markdown("---")
    st.markdown("**Detectable conditions:**")
    for c in CLASS_NAMES:
        st.markdown(f"- {c}")
    st.markdown("---")
    st.caption("Model: EfficientNetB3 · LLM: Phi-3-mini · XAI: Grad-CAM++")

uploaded = st.file_uploader(
    "Upload retinal fundus image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    # Load vision model
    with st.spinner("Loading vision model…"):
        vision_model = load_vision_model()

    # Predict
    with st.spinner("Analyzing image…"):
        disease, confidence, all_probs, arr = predict(img_pil, vision_model)

    # Layout
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.markdown('<div class="section-header">📷 Input Image</div>', unsafe_allow_html=True)
        st.image(img_pil, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🔬 Diagnosis Result</div>', unsafe_allow_html=True)
        color = "#27ae60" if disease == "Healthy" else "#e74c3c"
        st.markdown(
            f'<div class="disease-badge" style="background:{color};">{disease}</div>',
            unsafe_allow_html=True,
        )
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        st.markdown('<div class="section-header">📊 All Class Probabilities</div>', unsafe_allow_html=True)
        for name, prob in sorted(zip(CLASS_NAMES, all_probs), key=lambda x: -x[1]):
            bar_color = "#1a73e8" if name == disease else "#ddd"
            st.markdown(
                f'<div class="confidence-bar-label">{name} — {prob:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.progress(float(prob))

    # Grad-CAM++
    if use_gradcam:
        st.markdown("---")
        st.markdown('<div class="section-header">🔥 Grad-CAM++ Activation Map</div>', unsafe_allow_html=True)
        with st.spinner("Generating heatmap…"):
            try:
                orig_np, heatmap_np, overlay_np = make_gradcam_plusplus(arr, vision_model, LAST_CONV_LAYER)
                c1, c2, c3 = st.columns(3)
                c1.image(orig_np, caption="Original", use_container_width=True)
                c2.image(heatmap_np, caption="Heatmap", use_container_width=True)
                c3.image(overlay_np, caption="Overlay", use_container_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM++ failed: {e}")

    # LLM Report
    if use_llm:
        st.markdown("---")
        st.markdown('<div class="section-header">📝 AI Medical Report</div>', unsafe_allow_html=True)
        with st.spinner("Generating medical report (Phi-3)…"):
            try:
                tokenizer, llm = load_llm()
                report = generate_report(disease, confidence, tokenizer, llm)
                st.markdown(f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"LLM report failed: {e}")

    st.markdown(
        '<div class="warning-box">⚠️ This tool is for research purposes only. '
        'It is not a substitute for professional medical advice, diagnosis, or treatment. '
        'Always consult a qualified ophthalmologist.</div>',
        unsafe_allow_html=True,
    )

else:
    st.info("👆 Upload a retinal fundus image to begin diagnosis.")
