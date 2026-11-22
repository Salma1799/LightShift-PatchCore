"""Branchement des vrais modèles PatchCore (checkpoints anomalib 2.7.1).

Les checkpoints sont ceux produits par src/models/train_patchcore.py :
    results/Patchcore/MVTecAD/<category>/latest/weights/lightning/model.ckpt

⚠️ À VALIDER dès réception des checkpoints de Wissal : ce code suit l'API
anomalib 2.x (forward = pre_processor -> modèle -> post_processor), mais n'a
pas pu être exécuté sans les fichiers model.ckpt.

Méthode "lightshift" : en attente du membre 2. Quand ses briques seront prêtes,
brancher son code dans _predict_lightshift() sans changer le reste de l'API.
"""

import threading
from pathlib import Path

import cv2
import numpy as np

from app.models.base import Prediction


class PatchcoreModel:
    name = "patchcore"
    version = "patchcore-wrn50-l2l3-cs0.1-k9"

    def __init__(self, checkpoint_root: Path, categories: list[str]):
        self.checkpoint_root = Path(checkpoint_root)
        self.categories = categories
        self._models = {}
        self._lock = threading.Lock()

    def checkpoint_path(self, category: str) -> Path:
        return (self.checkpoint_root / category / "latest" / "weights"
                / "lightning" / "model.ckpt")

    def is_ready(self) -> bool:
        return all(self.checkpoint_path(c).exists() for c in self.categories)

    def default_threshold(self, category: str, method: str) -> float:
        # Le post-processor d'anomalib normalise les scores : le seuil appris vaut 0.5.
        # Rappel : ce seuil a été appris sur le test (voir guide de passation §5),
        # d'où l'intérêt de /calibrate.
        return 0.5

    def predict(self, image_rgb: np.ndarray, category: str, method: str) -> Prediction:
        if method == "lightshift":
            return self._predict_lightshift(image_rgb, category)
        return self._predict_baseline(image_rgb, category)

    # --- baseline -------------------------------------------------------------
    def _predict_baseline(self, image_rgb: np.ndarray, category: str) -> Prediction:
        import torch

        model = self._load(category)
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)

        score = float(_first(output, "pred_score").reshape(-1)[0])
        amap = _first(output, "anomaly_map").squeeze().cpu().numpy().astype(np.float32)
        h, w = image_rgb.shape[:2]
        if amap.shape != (h, w):
            amap = cv2.resize(amap, (w, h), interpolation=cv2.INTER_LINEAR)
        return Prediction(score=score, anomaly_map=amap)

    # --- méthode améliorée (membre 2) -------------------------------------------
    def _predict_lightshift(self, image_rgb: np.ndarray, category: str) -> Prediction:
        raise NotImplementedError(
            "La méthode LightShift (B1/B2) n'est pas encore livrée par le membre 2. "
            "Utiliser method=baseline, ou /calibrate pour appliquer B3."
        )

    def _load(self, category: str):
        with self._lock:
            if category not in self._models:
                path = self.checkpoint_path(category)
                if not path.exists():
                    raise FileNotFoundError(f"Checkpoint introuvable : {path}")
                from anomalib.models import Patchcore

                model = Patchcore.load_from_checkpoint(str(path), map_location="cpu")
                model.eval()
                self._models[category] = model
            return self._models[category]


def _first(output, name: str):
    value = getattr(output, name, None)
    if value is None and isinstance(output, dict):
        value = output.get(name)
    if value is None:
        raise RuntimeError(f"Sortie anomalib sans champ '{name}'.")
    return value
