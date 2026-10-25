"""Explainability helpers (Grad-CAM + SHAP). Reused by the Phase 5 script and the Phase 6 dashboard."""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.data import MEAN, STD
from src.model import build_model, gradcam_layer

_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def disable_inplace(model):
    """In-place activations break gradient hooks (Grad-CAM / SHAP); switch them off."""
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = build_model(ck["model"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    disable_inplace(model)
    return model, ck["classes"]


def pil_to_tensor(img, size=224):
    img = img.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return _tf(img).unsqueeze(0)


def denorm(x):
    """[3,H,W] normalised tensor -> HxWx3 float32 image in [0,1]."""
    m, s = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    return (x.detach().cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy().astype(np.float32)


@torch.no_grad()
def predict_proba(model, x):
    return torch.softmax(model(x), 1)


def gradcam(model, x, targets):
    """x: [N,3,H,W] on the model's device; targets: list of class ids. Returns [N,H,W] maps in [0,1]."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cam = GradCAM(model=model, target_layers=[gradcam_layer(model)])
    return cam(input_tensor=x, targets=[ClassifierOutputTarget(int(t)) for t in targets])


def cam_overlay(rgb01, cam):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    return show_cam_on_image(rgb01.astype(np.float32), cam, use_rgb=True)


def shap_maps(model, x, targets, background, nsamples=50):
    """Expected-gradients SHAP. Returns [N,3,H,W]: attribution of each input pixel/channel to the target class."""
    import shap
    sv = shap.GradientExplainer(model, background).shap_values(x, nsamples=nsamples)
    if isinstance(sv, list):  # older shap: list over outputs
        sv = np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    out = []
    for i, t in enumerate(targets):
        a = sv[i, ..., int(t)]
        if a.shape[0] != 3:  # channels-last safety
            a = a.transpose(2, 0, 1)
        out.append(a)
    return np.stack(out)


def shap_overlay(rgb01, smap, vmax=None):
    """Signed SHAP map (sum over channels) drawn over a faded image. Red = pushes toward the class."""
    import matplotlib
    s = smap.sum(0)
    vmax = vmax or (np.percentile(np.abs(s), 99) + 1e-12)
    heat = matplotlib.colormaps["bwr"]((np.clip(s / vmax, -1, 1) + 1) / 2)[..., :3]
    gray = np.repeat(rgb01.mean(2, keepdims=True), 3, 2) * 0.6 + 0.4
    a = np.clip(np.abs(s) / vmax, 0, 1)[..., None] * 0.8
    return (gray * (1 - a) + heat * a).astype(np.float32)
