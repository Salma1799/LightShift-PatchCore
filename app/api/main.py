"""API LightShift-PatchCore.

Lancer :  uvicorn app.api.main:app --reload
Docs   :  http://localhost:8000/docs
"""

import math
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app import config
from app.api.schemas import CalibrateResponse, ErrorResponse, HealthResponse, PredictResponse
from app.core.conformal import conformal_threshold, is_anomaly, min_calibration_size
from app.core.imaging import InvalidImageError, decode_image, encode_png_base64, heatmap_overlay
from app.core.lighting import apply_lighting_shift
from app.core.thresholds import ThresholdStore
from app.models.registry import build_model

ERRORS = {400: {"model": ErrorResponse}, 422: {"model": ErrorResponse},
          501: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.data = {"predict_requests": 0, "predict_errors": 0, "decisions": {"normal": 0, "anomaly": 0},
                     "calibrations": 0, "total_inference_ms": 0.0}

    def record_prediction(self, decision: str, ms: float):
        with self._lock:
            self.data["predict_requests"] += 1
            self.data["decisions"][decision] += 1
            self.data["total_inference_ms"] += ms

    def record_error(self):
        with self._lock:
            self.data["predict_errors"] += 1

    def record_calibration(self):
        with self._lock:
            self.data["calibrations"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            n = self.data["predict_requests"]
            out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.data.items()}
            out["mean_inference_ms"] = round(self.data["total_inference_ms"] / n, 2) if n else None
            return out


def create_app(backend: Optional[str] = None, state_dir: Optional[Path] = None) -> FastAPI:
    model = build_model(backend)
    store = ThresholdStore(state_dir or config.STATE_DIR)
    metrics = Metrics()

    api = FastAPI(
        title="LightShift-PatchCore API",
        version="0.1.0",
        description="Détection d'anomalies industrielles robuste aux changements d'éclairage.",
    )

    # ---------------------------------------------------------------- helpers
    def check_category(category: str):
        if category not in config.CATEGORIES:
            raise HTTPException(422, f"Catégorie inconnue '{category}'. Valeurs : {config.CATEGORIES}")

    def check_method(method: str):
        if method not in config.METHODS:
            raise HTTPException(422, f"Méthode inconnue '{method}'. Valeurs : {config.METHODS}")

    async def read_image(upload: UploadFile):
        data = await upload.read()
        if len(data) > config.MAX_IMAGE_BYTES:
            raise HTTPException(400, f"Image trop lourde ({len(data)} octets, max {config.MAX_IMAGE_BYTES}).")
        try:
            return decode_image(data)
        except InvalidImageError as exc:
            raise HTTPException(400, f"{upload.filename}: {exc}") from exc

    def current_threshold(category: str, method: str):
        saved = store.get(model.name, method, category)
        if saved is not None:
            return float(saved["threshold"]), "calibrated"
        return float(model.default_threshold(category, method)), "default"

    def run_model(image, category: str, method: str):
        if not model.is_ready():
            raise HTTPException(503, "Modèle non disponible (checkpoints manquants ?). Voir /health.")
        try:
            return model.predict(image, category, method)
        except NotImplementedError as exc:
            raise HTTPException(501, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    # ---------------------------------------------------------------- routes
    @api.get("/health", response_model=HealthResponse)
    def health():
        ready = model.is_ready()
        return HealthResponse(status="ok" if ready else "degraded", model_loaded=ready,
                              model_backend=model.name, model_version=model.version,
                              categories=config.CATEGORIES, methods=config.METHODS)

    @api.post("/predict", response_model=PredictResponse, responses=ERRORS)
    async def predict(
        image: UploadFile = File(..., description="Image PNG ou JPG"),
        category: str = Form(..., description="metal_nut, screw ou transistor"),
        method: str = Form("baseline", description="baseline ou lightshift"),
        shift: Optional[str] = Form(None, description="Optionnel : simuler un éclairage (démo)"),
        severity: int = Form(1, description="Niveau du changement d'éclairage : 1, 2 ou 3"),
    ):
        check_category(category)
        check_method(method)
        if shift is not None and shift not in config.SHIFTS:
            raise HTTPException(422, f"Shift inconnu '{shift}'. Valeurs : {config.SHIFTS}")
        if severity not in config.SEVERITIES:
            raise HTTPException(422, f"Sévérité invalide {severity}. Valeurs : {config.SEVERITIES}")

        rgb = await read_image(image)
        if shift is not None:
            rgb = apply_lighting_shift(rgb, shift, severity)

        start = time.perf_counter()
        try:
            pred = run_model(rgb, category, method)
        except HTTPException:
            metrics.record_error()
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000

        threshold, source = current_threshold(category, method)
        decision = "anomaly" if is_anomaly(pred.score, threshold) else "normal"
        metrics.record_prediction(decision, elapsed_ms)

        return PredictResponse(
            category=category, method=method, score=round(pred.score, 6),
            threshold=round(threshold, 6), threshold_source=source, decision=decision,
            anomaly_map=encode_png_base64(heatmap_overlay(rgb, pred.anomaly_map)),
            processed_image=encode_png_base64(rgb) if shift else None,
            applied_shift=shift, applied_severity=severity if shift else None,
            inference_time_ms=round(elapsed_ms, 2),
            model_backend=model.name, model_version=model.version,
        )

    @api.post("/calibrate", response_model=CalibrateResponse, responses=ERRORS)
    async def calibrate(
        images: list[UploadFile] = File(..., description="Images NORMALES du nouvel éclairage"),
        category: str = Form(...),
        alpha: float = Form(0.1, description="Taux de fausses alertes visé, ex. 0.1 = 10 %"),
        method: str = Form("baseline"),
    ):
        """Brique B3 : recalcule le seuil à partir d'images normales du nouvel éclairage.

        Ces images ne doivent PAS être celles utilisées ensuite pour évaluer le modèle.
        """
        check_category(category)
        check_method(method)
        if not 0 < alpha < 1:
            raise HTTPException(422, "alpha doit être strictement entre 0 et 1.")
        if len(images) > config.MAX_CALIBRATION_IMAGES:
            raise HTTPException(400, f"Trop d'images (max {config.MAX_CALIBRATION_IMAGES}).")

        n_min = min_calibration_size(alpha)
        if len(images) < n_min:
            raise HTTPException(
                422, f"{len(images)} image(s) ne suffisent pas pour alpha={alpha} : "
                     f"il en faut au moins {n_min}, sinon le seuil conforme est infini.")

        scores = []
        for upload in images:
            rgb = await read_image(upload)
            scores.append(run_model(rgb, category, method).score)

        new_threshold = conformal_threshold(scores, alpha)
        assert math.isfinite(new_threshold)  # garanti par le contrôle n_min ci-dessus
        previous, _ = current_threshold(category, method)
        entry = store.set(model.name, method, category, new_threshold, alpha, len(scores))
        metrics.record_calibration()

        return CalibrateResponse(
            category=category, method=method, n_images=len(scores), alpha=alpha,
            new_threshold=round(new_threshold, 6), previous_threshold=round(previous, 6),
            scores_min=round(min(scores), 6), scores_max=round(max(scores), 6),
            calibrated_at=entry["calibrated_at"],
            note=("Garantie : FPR <= alpha en moyenne si les futures images normales viennent du même "
                  "éclairage que ces images. Aucune garantie sur le rappel des défauts."),
        )

    @api.delete("/calibrate/{category}")
    def reset_calibration(category: str, method: str = "baseline"):
        check_category(category)
        check_method(method)
        removed = store.reset(model.name, method, category)
        threshold, source = current_threshold(category, method)
        return {"category": category, "method": method, "reset": removed,
                "threshold": threshold, "threshold_source": source}

    @api.get("/metrics")
    def get_metrics():
        snap = metrics.snapshot()
        snap["thresholds"] = {
            f"{m}:{c}": dict(zip(("threshold", "source"), current_threshold(c, m)))
            for m in config.METHODS for c in config.CATEGORIES
        }
        return snap

    return api


app = create_app()
