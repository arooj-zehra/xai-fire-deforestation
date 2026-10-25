"""Early-warning dashboard: upload a satellite tile -> class + confidence + Grad-CAM (+ optional SHAP).

Run from the repo root:   streamlit run app/app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.explain import (cam_overlay, denorm, gradcam, load_checkpoint, pil_to_tensor, predict_proba,
                         shap_maps, shap_overlay)

st.set_page_config(page_title="Wildfire & Deforestation Early Warning", layout="wide")
DATA = os.environ.get("TILE_DATA", "data/processed")
DEFAULT_CKPT = os.environ.get("CKPT", "outputs/B_aug/best.pt")


@st.cache_resource(show_spinner="Loading model...")
def get_model(path):
    return load_checkpoint(path, "cpu")


@st.cache_resource(show_spinner=False)
def get_background(n=20):
    p = os.path.join(DATA, "labels.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df = df[df["split"] == "train"].sample(n, random_state=0)
    return torch.cat([pil_to_tensor(Image.open(os.path.join(DATA, f))) for f in df["path"]])


def sample_tiles():
    p = os.path.join(DATA, "labels.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df[df["split"] == "test"].reset_index(drop=True)


def center_square(img):
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


st.title("Wildfire & Deforestation Early Warning")
st.caption("Upload a satellite tile. The model classifies it as Normal / Wildfire / Deforestation and shows "
           "which regions drove the decision.")

with st.sidebar:
    st.header("Settings")
    ckpt = st.text_input("Model checkpoint", DEFAULT_CKPT)
    threshold = st.slider("Alert confidence threshold", 0.30, 0.99, 0.50, 0.01,
                          help="Raise an alert when a non-Normal class is predicted with at least this confidence.")
    bg_ok = get_background() is not None if os.path.exists(os.path.join(DATA, "labels.csv")) else False
    use_shap = st.checkbox("Also compute SHAP (slow on CPU)", value=False, disabled=not bg_ok,
                           help="Needs data/processed/ for background tiles." if not bg_ok else None)
    st.markdown("---")
    st.caption("Trained on Planet Amazon + Kaggle Canada wildfire tiles at 224x224 px. Not validated on other sensors "
               "or regions. Kaggle 'wildfire' tiles are historical burn areas, so wildfire explanations are diffuse; "
               "cleared-forest explanations are the most reliable.")

if not os.path.exists(ckpt):
    st.error(f"Checkpoint not found: `{ckpt}`. Download `best.pt` from Drive (outputs/B_aug) into that path.")
    st.stop()
model, classes = get_model(ckpt)

img = None
up = st.file_uploader("Satellite tile (JPG/PNG)", type=["jpg", "jpeg", "png"])
if up is not None:
    img = Image.open(up).convert("RGB")
    src_label = up.name
tiles = sample_tiles()
if tiles is not None:
    with st.expander("...or try a random test tile"):
        want = st.selectbox("True class", ["any"] + classes)
        if st.button("Load random tile"):
            pool = tiles if want == "any" else tiles[tiles["label"] == want]
            st.session_state["sample"] = pool.sample(1).iloc[0].to_dict()
    if up is None and "sample" in st.session_state:
        s = st.session_state["sample"]
        img = Image.open(os.path.join(DATA, s["path"])).convert("RGB")
        src_label = f"sample tile (true class: {s['label']}, source: {s['source']})"

if img is None:
    st.info("Upload a tile to begin.")
    st.stop()

img = center_square(img)
x = pil_to_tensor(img)
probs = predict_proba(model, x)[0].numpy()
pred = int(probs.argmax())
conf = float(probs[pred])

if classes[pred] != "Normal" and conf >= threshold:
    st.error(f"ALERT: **{classes[pred]}** detected - confidence {conf:.0%}")
elif classes[pred] != "Normal":
    st.warning(f"Possible **{classes[pred]}** ({conf:.0%}) - below the alert threshold of {threshold:.0%}")
else:
    st.success(f"No alert - classified as **Normal** ({conf:.0%})")
st.caption(f"Input: {src_label}")

target = st.selectbox("Explain which class?", classes, index=pred)
t = classes.index(target)
rgb = denorm(x[0])
cam = gradcam(model, x, [t])[0]
cols = st.columns(3 if use_shap else 2)
cols[0].image(rgb, caption="Tile (resized to 224x224)", use_container_width=True)
cols[1].image(cam_overlay(rgb, cam), caption=f"Grad-CAM for '{target}' (red = most influential; coarse 7x7 grid)",
              use_container_width=True)
if use_shap:
    with st.spinner("Computing SHAP..."):
        sm = shap_maps(model, x, [t], get_background(), nsamples=50)[0]
    cols[2].image(shap_overlay(rgb, sm), caption=f"SHAP for '{target}' (red = supports, blue = opposes)",
                  use_container_width=True)

st.subheader("Class probabilities")
st.bar_chart(pd.Series(probs, index=classes, name="probability"))
