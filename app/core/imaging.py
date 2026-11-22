"""Décodage d'images, carte de chaleur et encodage base64."""

import base64

import cv2
import numpy as np


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Octets PNG/JPG -> image RGB uint8 (H, W, 3)."""
    if not data:
        raise InvalidImageError("Fichier vide.")
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise InvalidImageError("Le fichier n'est pas une image lisible (PNG ou JPG attendu).")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_png_base64(rgb: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Encodage PNG impossible.")
    return base64.b64encode(png.tobytes()).decode("ascii")


def heatmap_overlay(rgb: np.ndarray, anomaly_map: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    """Superpose la carte d'anomalie (normalisée 0..1 sur l'image) à l'image d'origine."""
    h, w = rgb.shape[:2]
    amap = anomaly_map.astype(np.float32)
    if amap.shape != (h, w):
        amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(amap.min()), float(amap.max())
    norm = (amap - lo) / (hi - lo) if hi > lo else np.zeros_like(amap)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - opacity, colored, opacity, 0)
