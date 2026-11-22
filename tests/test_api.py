import base64

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app


def png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def normal_image(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), 115, dtype=np.float32)
    base += rng.normal(0, 3, size=base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


def defect_image() -> np.ndarray:
    img = normal_image(99)
    cv2.circle(img, (64, 64), 10, (255, 255, 255), -1)
    return img


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(backend="mock", state_dir=tmp_path))


def post_predict(client, img, **form):
    data = {"category": "screw", **form}
    return client.post("/predict", data=data,
                       files={"image": ("x.png", png_bytes(img), "image/png")})


# ------------------------------------------------------------------ /health
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_backend"] == "mock"


# ------------------------------------------------------------------ /predict
def test_predict_output_contract(client):
    r = post_predict(client, normal_image())
    assert r.status_code == 200
    body = r.json()
    for key in ("score", "threshold", "decision", "anomaly_map", "inference_time_ms",
                "model_version", "threshold_source"):
        assert key in body
    assert body["decision"] in ("normal", "anomaly")
    assert body["threshold_source"] == "default"
    heat = cv2.imdecode(np.frombuffer(base64.b64decode(body["anomaly_map"]), np.uint8),
                        cv2.IMREAD_COLOR)
    assert heat.shape[:2] == (128, 128)


def test_predict_is_deterministic(client):
    a = post_predict(client, normal_image()).json()["score"]
    b = post_predict(client, normal_image()).json()["score"]
    assert a == b


def test_mock_scores_defect_higher_than_normal(client):
    normal = post_predict(client, normal_image()).json()
    defect = post_predict(client, defect_image()).json()
    assert defect["score"] > normal["score"]
    assert normal["decision"] == "normal" and defect["decision"] == "anomaly"


def test_predict_with_lighting_shift_returns_processed_image(client):
    r = post_predict(client, normal_image(), shift="brightness", severity="3")
    body = r.json()
    assert r.status_code == 200
    assert body["applied_shift"] == "brightness" and body["applied_severity"] == 3
    assert body["processed_image"]


@pytest.mark.parametrize("form,code", [
    ({"category": "banana"}, 422),
    ({"method": "magic"}, 422),
    ({"shift": "fog"}, 422),
    ({"shift": "gamma", "severity": "7"}, 422),
])
def test_predict_rejects_bad_parameters(client, form, code):
    assert post_predict(client, normal_image(), **form).status_code == code


def test_predict_rejects_non_image(client):
    r = client.post("/predict", data={"category": "screw"},
                    files={"image": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_predict_rejects_empty_file(client):
    r = client.post("/predict", data={"category": "screw"},
                    files={"image": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_predict_requires_image(client):
    assert client.post("/predict", data={"category": "screw"}).status_code == 422


# ------------------------------------------------------------------ /calibrate
def calib_files(n, shift_brightness=0):
    files = []
    for i in range(n):
        img = np.clip(normal_image(i).astype(int) + shift_brightness, 0, 255).astype(np.uint8)
        files.append(("images", (f"n{i}.png", png_bytes(img), "image/png")))
    return files


def test_calibrate_updates_threshold(client):
    r = client.post("/calibrate", data={"category": "screw", "alpha": "0.1"},
                    files=calib_files(20, shift_brightness=-70))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_images"] == 20
    assert body["previous_threshold"] == 0.5
    assert body["scores_min"] <= body["new_threshold"] <= body["scores_max"]

    pred = post_predict(client, normal_image()).json()
    assert pred["threshold_source"] == "calibrated"
    assert pred["threshold"] == body["new_threshold"]


def test_calibration_reduces_false_alarms_under_lighting_shift(client):
    """Scénario de la démo : sous un éclairage sombre, le seuil par défaut déclenche
    de fausses alertes ; après /calibrate sur d'autres images normales, elles baissent."""
    dark = lambda i: np.clip(normal_image(i).astype(int) - 70, 0, 255).astype(np.uint8)
    test_ids = range(100, 130)
    before = sum(post_predict(client, dark(i)).json()["decision"] == "anomaly" for i in test_ids)

    client.post("/calibrate", data={"category": "screw", "alpha": "0.1"},
                files=calib_files(20, shift_brightness=-70))
    after = sum(post_predict(client, dark(i)).json()["decision"] == "anomaly" for i in test_ids)
    assert before > after


def test_calibrate_rejects_too_few_images(client):
    r = client.post("/calibrate", data={"category": "screw", "alpha": "0.05"},
                    files=calib_files(10))
    assert r.status_code == 422
    assert "19" in r.json()["detail"]


def test_calibrate_rejects_bad_alpha(client):
    r = client.post("/calibrate", data={"category": "screw", "alpha": "1.5"},
                    files=calib_files(10))
    assert r.status_code == 422


def test_calibration_is_per_category_and_resettable(client):
    client.post("/calibrate", data={"category": "screw", "alpha": "0.1"}, files=calib_files(10))
    other = client.post("/predict", data={"category": "metal_nut"},
                        files={"image": ("x.png", png_bytes(normal_image()), "image/png")}).json()
    assert other["threshold_source"] == "default"

    r = client.delete("/calibrate/screw")
    assert r.status_code == 200 and r.json()["reset"] is True
    assert post_predict(client, normal_image()).json()["threshold_source"] == "default"


def test_calibration_persists_across_restart(tmp_path):
    c1 = TestClient(create_app(backend="mock", state_dir=tmp_path))
    c1.post("/calibrate", data={"category": "screw", "alpha": "0.1"}, files=calib_files(10))
    c2 = TestClient(create_app(backend="mock", state_dir=tmp_path))
    assert post_predict(c2, normal_image()).json()["threshold_source"] == "calibrated"


# ------------------------------------------------------------------ /metrics
def test_metrics_counts_requests(client):
    post_predict(client, normal_image())
    post_predict(client, defect_image())
    m = client.get("/metrics").json()
    assert m["predict_requests"] == 2
    assert m["decisions"] == {"normal": 1, "anomaly": 1}
    assert "baseline:screw" in m["thresholds"]


# ------------------------------------------------------------------ backend patchcore
def test_patchcore_backend_without_checkpoints_is_degraded(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.CHECKPOINT_ROOT", tmp_path / "missing")
    c = TestClient(create_app(backend="patchcore", state_dir=tmp_path))
    assert c.get("/health").json()["status"] == "degraded"
    assert post_predict(c, normal_image()).status_code == 503
