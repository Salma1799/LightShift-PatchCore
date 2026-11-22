"""Faux modèle pour développer l'application sans attendre les checkpoints.

⚠️ Ce n'est PAS PatchCore et ses scores n'ont aucune valeur scientifique.
Il imite seulement le comportement utile pour tester l'application :
  - il est déterministe (même image -> même score), donc testable ;
  - il réagit aux zones localement inhabituelles (carte de chaleur plausible) ;
  - en méthode "baseline", son score monte quand l'éclairage s'écarte de la normale,
    comme PatchCore sous changement d'éclairage ;
  - en méthode "lightshift", il normalise d'abord l'image par canal (idée de B2),
    donc il est moins sensible à l'éclairage.
"""

import cv2
import numpy as np

from app.models.base import Prediction


class MockModel:
    name = "mock"
    version = "mock-0.1"

    def is_ready(self) -> bool:
        return True

    def default_threshold(self, category: str, method: str) -> float:
        return 0.5

    def predict(self, image_rgb: np.ndarray, category: str, method: str) -> Prediction:
        img = image_rgb.astype(np.float32) / 255.0

        if method == "lightshift":
            mean = img.mean(axis=(0, 1), keepdims=True)
            std = img.std(axis=(0, 1), keepdims=True) + 1e-6
            img = np.clip((img - mean) / std * 0.2 + 0.5, 0.0, 1.0)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        background = cv2.GaussianBlur(gray, (0, 0), sigmaX=8)
        local = cv2.GaussianBlur(np.abs(gray - background), (0, 0), sigmaX=4)

        local_term = float(np.percentile(local, 99.5)) * 4.0
        lighting_term = abs(float(gray.mean()) - 0.45) * 1.2
        score = float(np.clip(0.15 + local_term + lighting_term, 0.0, 1.0))

        return Prediction(score=score, anomaly_map=local)
