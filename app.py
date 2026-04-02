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
# UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-block">
    <p class="header-title">👁 Eye Disease AI Diagnostics</p>
    <p class="header-sub">EfficientNetB3 · Grad-CAM++ · Phi-3 Medical Report · 7 Disease Classes</p>
</div>
""", unsafe_allow_html=True)
 
# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    show_gradcam = st.toggle("Show Grad-CAM++ Heatmap", value=True)
    show_all_probs = st.toggle("Show All Class Probabilities", value=False)
    use_llm = st.toggle("Use Phi-3 LLM for Report", value=False,
                        help="Slower but more dynamic explanations. Requires GPU for best performance.")
 
    st.markdown("---")
    st.markdown("**📋 Supported Conditions**")
    for name in CLASS_NAMES:
        severity, color = DISEASE_INFO[name]
        st.markdown(f"<span style='color:{color};font-size:0.8rem'>● {name}</span>", unsafe_allow_html=True)
 
    st.markdown("---")
    st.caption("Model: EfficientNetB3 · Input: 300×300px")
    st.caption("Place `best_efficientnetb3.h5` in the app directory.")
 
# Load models
vision_model = load_vision_model()
 
tokenizer, llm = None, None
if use_llm:
    with st.spinner("Loading Phi-3 LLM..."):
        tokenizer, llm = load_llm()
 
# Main area
col1, col2 = st.columns([1, 1.3], gap="large")
 
with col1:
    st.markdown("#### 🖼 Upload Retinal Image")
    uploaded_file = st.file_uploader(
        "JPG, PNG, JPEG supported",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
 
    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)
    else:
        st.info("Upload a retinal scan image to begin diagnosis.")
        analyze_btn = False
 
with col2:
    if uploaded_file and analyze_btn:
        if vision_model is None:
            st.error("Vision model not loaded. Please check `best_efficientnetb3.h5`.")
        else:
            with st.spinner("Running EfficientNetB3 inference..."):
                img_array = preprocess_image(pil_img)
                disease, confidence, all_probs = predict_disease(img_array, vision_model)
 
            severity_label, severity_color = DISEASE_INFO.get(disease, ('Unknown', '#94a3b8'))
 
            st.markdown("#### 🩺 Diagnosis Result")
            st.markdown(f"""
<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <span style="color:#94a3b8;font-size:0.85rem">PREDICTED CONDITION</span>
        <span style="color:{severity_color};font-size:0.82rem;background:rgba(0,0,0,0.3);padding:2px 8px;border-radius:100px;border:1px solid {severity_color}40">{severity_label}</span>
    </div>
    <div class="disease-badge">{disease}</div>
    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
        <span style="color:#94a3b8;font-size:0.8rem">Confidence</span>
        <span style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:0.85rem">{confidence:.1%}</span>
    </div>
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{confidence*100:.1f}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)
 
            if show_all_probs:
                st.markdown("**All Class Probabilities**")
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                for cls, prob in sorted_probs:
                    bar_w = int(prob * 100)
                    highlight = "color:#00d4ff;" if cls == disease else ""
                    st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
    <span style="font-size:0.78rem;{highlight}min-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{cls}</span>
    <div style="background:#1e293b;border-radius:100px;height:6px;flex:1;overflow:hidden">
        <div style="width:{bar_w}%;height:100%;background:{'#00d4ff' if cls == disease else '#475569'};border-radius:100px"></div>
    </div>
    <span style="font-family:'Space Mono',monospace;font-size:0.75rem;{highlight}min-width:40px;text-align:right">{prob:.1%}</span>
</div>
""", unsafe_allow_html=True)
 
            # Medical Report
            st.markdown("#### 📋 AI Medical Report")
            with st.spinner("Generating medical explanation..."):
                report_lines = llm_explain(disease, confidence, tokenizer, llm)
 
            report_html = ""
            for i, line in enumerate(report_lines, 1):
                # Strip leading number if present
                text = line.lstrip("0123456789. ").strip()
                report_html += f"""
<div class="report-line">
    <span class="line-num">{i}.</span>
    <span class="line-text">{text}</span>
</div>"""
 
            st.markdown(f'<div class="card">{report_html}</div>', unsafe_allow_html=True)
 
            # Grad-CAM
            if show_gradcam:
                st.markdown("#### 🔥 Grad-CAM++ Visualization")
                with st.spinner("Computing saliency map..."):
                    heatmap = generate_gradcam(img_array, vision_model)
                if heatmap is not None:
                    buf = build_gradcam_figure(pil_img, heatmap)
                    st.image(buf, use_container_width=True)
                    st.caption("Highlighted regions show areas the model focused on for the prediction.")
 
    elif not uploaded_file:
        st.markdown("""
<div class="card" style="text-align:center;padding:3rem 1.5rem;border:1px dashed rgba(255,255,255,0.08)">
    <div style="font-size:3rem;margin-bottom:1rem">👁</div>
    <div style="color:#94a3b8;font-size:0.95rem">Results will appear here after analysis</div>
    <div style="color:#475569;font-size:0.8rem;margin-top:0.5rem">Upload an image and click Analyze</div>
</div>
""", unsafe_allow_html=True)
