"""
PlantMD Pro — Streamlit Frontend
Professional plant disease detection UI with Grad-CAM visualization.
"""
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlantMD Pro",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0f0a;
    color: #e8f5e8;
}

.stApp { background-color: #0a0f0a; }

.main-header {
    background: linear-gradient(135deg, #0d1f0d, #060d06);
    border: 1px solid #1a3a1a;
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
}

.metric-card {
    background: #0d140d;
    border: 1px solid #1a2a1a;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.prediction-card {
    background: #0d140d;
    border-radius: 12px;
    padding: 24px;
    margin-top: 16px;
}

.healthy-badge {
    background: #166534;
    color: #86efac;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 700;
    display: inline-block;
}

.disease-badge {
    background: #7f1d1d;
    color: #fca5a5;
    padding: 4px 16px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 700;
    display: inline-block;
}

.severity-high { color: #ef4444; font-weight: 700; }
.severity-medium { color: #f97316; font-weight: 700; }
.severity-low { color: #eab308; font-weight: 700; }
.severity-none { color: #22c55e; font-weight: 700; }

.treatment-box {
    background: #060d06;
    border: 1px solid #1a3a1a;
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
}

.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 12px 32px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
    transform: translateY(-1px) !important;
}

.stFileUploader {
    border: 2px dashed #1a3a1a !important;
    border-radius: 12px !important;
    background: #0d140d !important;
}

div[data-testid="stSidebar"] {
    background-color: #060d06 !important;
    border-right: 1px solid #1a2a1a !important;
}

.info-chip {
    background: #1a3a1a;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    color: #4ade80;
    display: inline-block;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def call_predict_api(
    image_bytes: bytes,
    filename: str,
    generate_heatmap: bool,
    api_key: Optional[str] = None,
) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(
        f"{API_BASE}/api/v1/predict",
        files={"file": (filename, image_bytes, "image/jpeg")},
        data={"generate_heatmap": str(generate_heatmap).lower()},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_health() -> dict:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        return resp.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False}


def severity_class(s: str) -> str:
    return {"HIGH": "severity-high", "MEDIUM": "severity-medium", "LOW": "severity-low", "NONE": "severity-none"}.get(s, "severity-none")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 PlantMD Pro")
    st.markdown("---")

    health = get_health()
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    st.markdown(f"**API Status:** {status_color} `{health.get('status', 'unknown')}`")
    st.markdown(f"**Model:** `{'✅ Loaded' if health.get('model_loaded') else '❌ Not Loaded'}`")
    st.markdown(f"**Version:** `{health.get('version', 'N/A')}`")

    st.markdown("---")
    generate_heatmap = st.toggle("🔬 Generate Grad-CAM Heatmap", value=False, help="Visualize which leaf regions influenced the prediction")
    api_key = st.text_input("🔑 API Token (optional)", type="password", help="JWT token for authenticated requests")

    st.markdown("---")
    st.markdown("### 📚 Supported Crops")
    crops = ["🍅 Tomato", "🥔 Potato", "🌶️ Pepper", "🌽 Corn", "🍎 Apple", "🍇 Grape", "🌱 Healthy Plants"]
    for c in crops:
        st.markdown(f"- {c}")

    st.markdown("---")
    st.markdown('<div class="info-chip">EfficientNetV2-S</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-chip">~95% Accuracy</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-chip">38 Classes</div>', unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="font-size:2.5rem;font-weight:700;margin:0;color:#4ade80;">🌿 PlantMD Pro</h1>
    <p style="color:#4a6a4a;margin-top:8px;font-size:1rem;">
        AI-powered plant disease detection — upload a leaf image to get an instant diagnosis
    </p>
</div>
""", unsafe_allow_html=True)

# Stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", "EfficientNetV2-S", "Transfer Learning")
with col2:
    st.metric("Accuracy", "~95%", "Top-1")
with col3:
    st.metric("Disease Classes", "38", "PlantVillage")
with col4:
    st.metric("Inference", "< 100ms", "GPU accelerated")

st.markdown("---")

# ── Upload area ───────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("### 📷 Upload Leaf Image")
    uploaded = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        help="Max 10MB. Supported: JPG, PNG, WebP",
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption=f"📁 {uploaded.name}", use_column_width=True)
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:12px;color:#4a6a4a;margin-top:8px;">
            Size: {uploaded.size / 1024:.1f} KB &nbsp;|&nbsp;
            Dimensions: {image.width}×{image.height}px &nbsp;|&nbsp;
            Format: {image.format or 'N/A'}
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("🔍 Analyze Disease", use_container_width=True)
    else:
        st.info("👆 Upload a clear, well-lit leaf image for best results")
        predict_btn = False

with col_result:
    st.markdown("### 🧪 Analysis Result")

    if uploaded and predict_btn:
        with st.spinner("Running deep learning inference..."):
            try:
                t0 = time.time()
                image_bytes = pil_to_bytes(image.convert("RGB"))
                result = call_predict_api(
                    image_bytes,
                    uploaded.name,
                    generate_heatmap,
                    api_key or None,
                )
                elapsed = (time.time() - t0) * 1000

                # Store in session
                st.session_state["last_result"] = result
                st.session_state["last_image"] = image

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API server. Start the backend first:\n```\nuvicorn main:app --reload\n```")
                result = None
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API Error: {e.response.json().get('message', str(e))}")
                result = None
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                result = None

    result = st.session_state.get("last_result")

    if result:
        is_healthy = result.get("is_healthy", False)
        badge = '<span class="healthy-badge">✅ HEALTHY</span>' if is_healthy else '<span class="disease-badge">⚠️ DISEASE DETECTED</span>'
        severity = result.get("severity", "NONE")
        sev_cls = severity_class(severity)
        display_name = result.get('display_name') or result.get('class_key', 'Unknown')
        confidence = result.get('confidence', 0)
        affected_crop = result.get('affected_crop', 'N/A')
        inference_ms = result.get('inference_ms', 0)

        st.markdown(f"""
        <div class="prediction-card" style="border:1px solid {'#166534' if is_healthy else '#7f1d1d'};">
            <div style="margin-bottom:16px;">{badge}</div>
            <h2 style="font-size:1.5rem;font-weight:700;color:{'#4ade80' if is_healthy else '#fca5a5'};margin:8px 0;">
                {display_name}
            </h2>
            <div style="display:flex;gap:24px;margin:12px 0;flex-wrap:wrap;">
                <div>
                    <span style="color:#4a6a4a;font-size:12px;">CONFIDENCE</span><br>
                    <span style="font-size:1.4rem;font-weight:700;color:#4ade80;">{confidence*100:.1f}%</span>
                </div>
                <div>
                    <span style="color:#4a6a4a;font-size:12px;">SEVERITY</span><br>
                    <span class="{sev_cls}" style="font-size:1.1rem;">{severity}</span>
                </div>
                <div>
                    <span style="color:#4a6a4a;font-size:12px;">CROP</span><br>
                    <span style="font-size:1rem;color:#e8f5e8;">{affected_crop}</span>
                </div>
                <div>
                    <span style="color:#4a6a4a;font-size:12px;">INFERENCE</span><br>
                    <span style="font-size:0.9rem;font-family:'DM Mono',monospace;color:#60a5fa;">{inference_ms:.0f}ms</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.progress(float(confidence))

        # Treatment
        if not is_healthy:
            treatment = result.get("treatment", {})
            st.markdown("""
            <div class="treatment-box">
                <h4 style="color:#4ade80;margin-bottom:12px;">💊 Treatment Plan</h4>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**⚡ Immediate Action:** {treatment.get('immediate_action', 'N/A')}")
            st.markdown(f"**🧪 Fungicide:** {treatment.get('fungicide', 'N/A')}")
            st.markdown(f"**🛡️ Prevention:** {treatment.get('prevention', 'N/A')}")

        # Top 3
        top3 = result.get("top3", [])
        if top3:
            st.markdown("#### 📊 Top-3 Predictions")
            for pred in top3:
                pct = pred["confidence"] * 100
                st.markdown(f"**{pred['display_name']}** — {pct:.1f}%")
                st.progress(pred["confidence"])

        # Grad-CAM
        heatmap_b64 = result.get("heatmap_base64")
        if heatmap_b64:
            st.markdown("#### 🔬 Grad-CAM Explainability")
            st.caption("Highlighted regions show which parts of the leaf influenced the prediction most")
            heatmap_bytes = base64.b64decode(heatmap_b64)
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
            st.image(heatmap_img, caption="Grad-CAM Heatmap Overlay", use_column_width=True)

        # Raw JSON
        with st.expander("🔧 Raw API Response (JSON)"):
            display = {k: v for k, v in result.items() if k != "heatmap_base64"}
            st.json(display)

        if result.get("demo_mode"):
            st.warning("""
⚠️ **Demo Mode** — No trained model found yet.
Results shown are **random simulations** to demonstrate the UI pipeline.
To get real predictions, train the model:
```
python ml_pipeline/train.py --dataset /path/to/plantvillage --output models/
```
""")

    else:
        st.markdown("""
        <div style="border:2px dashed #1a3a1a;border-radius:12px;padding:60px;text-align:center;color:#2a4a2a;">
            <div style="font-size:48px;margin-bottom:16px;">🌿</div>
            <p style="font-size:1rem;">Upload an image and click <strong>Analyze Disease</strong> to see results</p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#2a4a2a;font-size:12px;font-family:'DM Mono',monospace;">
    PlantMD Pro v2.0.0 | Built with EfficientNetV2 + FastAPI + Streamlit | PlantVillage Dataset
</div>
""", unsafe_allow_html=True)