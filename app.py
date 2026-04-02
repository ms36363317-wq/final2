import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Eye Disease AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1c2536;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --danger: #ef4444;
    --success: #10b981;
    --warning: #f59e0b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background: var(--bg); }

/* Header */
.header-block {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.header-block::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(0,212,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    margin: 0;
}
.header-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

/* Disease badge */
.disease-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00d4ff22, #7c3aed22);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    color: var(--accent);
    margin: 0.5rem 0 1rem;
}

/* Confidence bar */
.conf-bar-bg {
    background: #1e293b;
    border-radius: 100px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin: 0.3rem 0 1rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    transition: width 0.8s ease;
}

/* Report lines */
.report-line {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    margin-bottom: 0.7rem;
    padding: 0.6rem 0.8rem;
    background: rgba(0,212,255,0.04);
    border-left: 2px solid var(--accent2);
    border-radius: 0 6px 6px 0;
}
.line-num {
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    font-size: 0.8rem;
    min-width: 1.4rem;
    margin-top: 2px;
}
.line-text {
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Upload area */
.stFileUploader > div {
    background: var(--surface2) !important;
    border: 2px dashed rgba(0,212,255,0.3) !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
CLASS_NAMES = [
    'Diabetic Retinopathy',
    'Disc Edema',
    'Healthy',
    'Myopia',
    'Pterygium',
    'Retinal Detachment',
    'Retinitis Pigmentosa'
]

IMG_SIZE = (300, 300)
LAST_CONV_LAYER = "top_conv"

DISEASE_INFO = {
    'Diabetic Retinopathy': ('⚠️ High', '#ef4444'),
    'Disc Edema':           ('⚠️ High', '#ef4444'),
    'Healthy':              ('✅ Normal', '#10b981'),
    'Myopia':               ('🟡 Moderate', '#f59e0b'),
    'Pterygium':            ('🟡 Moderate', '#f59e0b'),
    'Retinal Detachment':   ('🚨 Critical', '#dc2626'),
    'Retinitis Pigmentosa': ('⚠️ High', '#ef4444'),
}

# ─────────────────────────────────────────────
# Model Loaders
# ─────────────────────────────────────────────
@st.cache_resource
def load_vision_model():
    import tensorflow as tf
    from tensorflow import keras

    # Support both naming conventions
    for model_path in ["best_efficientnetb3.h5", "model.h5"]:
        if os.path.exists(model_path):
            break
    else:
        st.error("❌ Model file not found.\n\nPlace `best_efficientnetb3.h5` (or `model.h5`) in the same directory as this app.")
        return None

    NUM_CLASSES = len(CLASS_NAMES)

    # Strategy 1: Full model load (architecture + weights)
    try:
        return load_model(model_path, compile=False)
    except Exception:
        pass

    # Strategy 2: tf_keras full load
    try:
        import tf_keras
        return tf_keras.models.load_model(model_path, compile=False)
    except Exception:
        pass

    # Strategy 3: Weights-only file — rebuild EfficientNetB3 architecture then load weights
    try:
        base_model = keras.applications.EfficientNetB3(
            include_top=False,
            weights=None,
            input_shape=(300, 300, 3)
        )
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(256, activation="relu",
                               kernel_initializer="he_normal",
                               kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = keras.layers.Dropout(0.5)(x)
        output = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs=base_model.input, outputs=output)
        model.load_weights(model_path)
        return model
    except Exception as e1:
        pass

    # Strategy 4: Same architecture but try by_name=True
    try:
        base_model = keras.applications.EfficientNetB3(
            include_top=False, weights=None, input_shape=(300, 300, 3)
        )
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        output = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs=base_model.input, outputs=output)
        model.load_weights(model_path, by_name=True, skip_mismatch=True)
        return model
    except Exception as e2:
        st.error(f"❌ Could not load model.\n\nStrategy 3 error: {str(e1)[:300]}\n\nStrategy 4 error: {str(e2)[:300]}")
        return None

@st.cache_resource
def load_llm():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        return tokenizer, llm
    except Exception as e:
        st.warning(f"⚠️ Could not load Phi-3: {e}\n\nFalling back to rule-based explanations.")
        return None, None

# ─────────────────────────────────────────────
# Inference Functions
# ─────────────────────────────────────────────
def preprocess_image(pil_img):
    img = pil_img.resize(IMG_SIZE)
    img_array = np.array(img.convert("RGB")).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

def predict_disease(img_array, model):
    preds = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    all_probs = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
    return CLASS_NAMES[class_idx], confidence, all_probs

def generate_gradcam(img_array, model):
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]

        grads_sq = tf.square(grads)
        grads_cu = grads_sq * grads
        denom = 2 * grads_sq + tf.reduce_sum(conv_outputs * grads_cu, axis=(0, 1), keepdims=True)
        denom = tf.where(denom != 0, denom, tf.ones_like(denom))
        alphas = grads_sq / denom
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
        heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = (heatmap / (tf.reduce_max(heatmap) + 1e-8)).numpy()
        return heatmap
    except Exception as e:
        st.warning(f"Grad-CAM skipped: {e}")
        return None

def build_gradcam_figure(original_pil, heatmap):
    original = np.array(original_pil.convert("RGB").resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    heatmap_resized = np.maximum(heatmap_resized, 0)
    heatmap_resized /= np.max(heatmap_resized) + 1e-8
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    heatmap_u8 = cv2.GaussianBlur(heatmap_u8, (31, 31), 0)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.65, heatmap_color, 0.35, 0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#111827')
    titles = ["Original", "Grad-CAM++ Heatmap", "Overlay"]
    imgs = [original, heatmap_resized, overlay]
    cmaps = [None, 'jet', None]
    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color='#94a3b8', fontsize=10, pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#111827')
    plt.close()
    buf.seek(0)
    return buf

def llm_explain(disease, confidence, tokenizer, llm):
    if llm is None:
        return _fallback_explain(disease, confidence)

    prompt = f"""You are an ophthalmology AI assistant.
Write exactly 5 concise medical sentences about this eye scan prediction.

Prediction: {disease}
Confidence: {confidence:.0%}

Format (5 lines only, no titles, no extra text):
1. State the prediction and confidence.
2. Brief definition of the condition.
3. Common symptoms the patient may experience.
4. Severity level and urgency.
5. Recommended next step for the patient.
"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.2
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = text.replace(prompt, "").strip()
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    return lines[:5]

def _fallback_explain(disease, confidence):
    info = {
        'Diabetic Retinopathy': [
            f"1. The model detected Diabetic Retinopathy with {confidence:.0%} confidence.",
            "2. Diabetic Retinopathy is retinal damage caused by poorly controlled blood sugar.",
            "3. Symptoms include blurred vision, floaters, and dark areas in sight.",
            "4. Severity: High — can lead to blindness if untreated.",
            "5. Consult an ophthalmologist urgently for retinal evaluation and treatment."
        ],
        'Disc Edema': [
            f"1. The model detected Disc Edema with {confidence:.0%} confidence.",
            "2. Disc Edema (papilledema) is swelling of the optic nerve head.",
            "3. Symptoms include headaches, visual disturbances, and nausea.",
            "4. Severity: High — may indicate elevated intracranial pressure.",
            "5. Urgent neurological and ophthalmological evaluation is needed."
        ],
        'Healthy': [
            f"1. The model found no signs of disease with {confidence:.0%} confidence.",
            "2. A healthy retina shows normal optic disc and blood vessel patterns.",
            "3. No abnormal symptoms expected.",
            "4. Severity: Normal — no immediate concern.",
            "5. Continue routine annual eye check-ups."
        ],
        'Myopia': [
            f"1. The model detected Myopia (nearsightedness) with {confidence:.0%} confidence.",
            "2. Myopia is a refractive error causing difficulty seeing distant objects.",
            "3. Symptoms include blurred distance vision and eye strain.",
            "4. Severity: Moderate — manageable with corrective lenses.",
            "5. Visit an optometrist for prescription glasses or contact lenses."
        ],
        'Pterygium': [
            f"1. The model detected Pterygium with {confidence:.0%} confidence.",
            "2. Pterygium is a fleshy growth on the conjunctiva that may extend onto the cornea.",
            "3. Symptoms include redness, irritation, and blurred vision if it covers the pupil.",
            "4. Severity: Moderate — surgical removal may be needed if vision is affected.",
            "5. See an ophthalmologist to monitor growth and discuss treatment options."
        ],
        'Retinal Detachment': [
            f"1. The model detected Retinal Detachment with {confidence:.0%} confidence.",
            "2. Retinal Detachment occurs when the retina separates from the eye wall.",
            "3. Symptoms include sudden flashes of light, floaters, and shadow in vision.",
            "4. Severity: CRITICAL — a medical emergency requiring immediate treatment.",
            "5. Go to an emergency eye clinic IMMEDIATELY — delay can cause permanent blindness."
        ],
        'Retinitis Pigmentosa': [
            f"1. The model detected Retinitis Pigmentosa with {confidence:.0%} confidence.",
            "2. Retinitis Pigmentosa is a genetic disorder causing progressive retinal degeneration.",
            "3. Symptoms include night blindness and tunnel vision worsening over time.",
            "4. Severity: High — currently no cure, but progression can be managed.",
            "5. Consult a retinal specialist for genetic counseling and management strategies."
        ],
    }
    return info.get(disease, [
        f"1. Prediction: {disease} ({confidence:.0%} confidence).",
        "2. Condition detected by AI analysis.",
        "3. Consult a specialist for proper evaluation.",
        "4. Severity to be determined by clinician.",
        "5. Schedule an appointment with an ophthalmologist."
    ])

# ─────────────────────────────────────────────
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
