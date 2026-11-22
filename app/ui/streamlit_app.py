"""Interface LightShift-PatchCore.

Lancer (l'API doit tourner) :  streamlit run app/ui/streamlit_app.py
"""

import base64
import math
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CATEGORIES = ["metal_nut", "screw", "transistor"]
SHIFT_LABELS = {
    "brightness": "Luminosité (plus sombre)",
    "contrast": "Contraste (plus faible)",
    "gamma": "Gamma",
    "color_temperature": "Température de couleur (plus jaune)",
    "shading": "Ombrage (gradient)",
}
METHOD_LABELS = {"baseline": "PatchCore (baseline)", "lightshift": "LightShift (méthode améliorée)"}

st.set_page_config(page_title="LightShift-PatchCore", page_icon="🔍", layout="wide")


# ------------------------------------------------------------------ helpers
def api_get(path):
    return requests.get(f"{API_URL}{path}", timeout=10)


def api_predict(file_bytes, filename, category, method, shift=None, severity=1):
    data = {"category": category, "method": method, "severity": str(severity)}
    if shift:
        data["shift"] = shift
    return requests.post(f"{API_URL}/predict", data=data,
                         files={"image": (filename, file_bytes)}, timeout=120)


def b64_image(b64):
    return base64.b64decode(b64)


def error_message(resp):
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


def show_result(result, title=None):
    if title:
        st.markdown(f"**{title}**")
    if result["decision"] == "anomaly":
        st.error(f"❌ DÉFAUT DÉTECTÉ — score {result['score']:.3f} > seuil {result['threshold']:.3f}")
    else:
        st.success(f"✅ PIÈCE NORMALE — score {result['score']:.3f} ≤ seuil {result['threshold']:.3f}")
    st.image(b64_image(result["anomaly_map"]), caption="Carte d'anomalie (rouge = suspect)",
             width="stretch")
    src = "recalibré" if result["threshold_source"] == "calibrated" else "par défaut"
    st.caption(f"Seuil {src} · {result['inference_time_ms']:.0f} ms · {result['model_version']}")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔍 LightShift-PatchCore")
st.sidebar.caption("Détection de défauts robuste aux changements d'éclairage")

try:
    health = api_get("/health").json()
    if health["status"] == "ok":
        st.sidebar.success(f"API connectée · modèle **{health['model_backend']}**")
    else:
        st.sidebar.warning("API connectée mais modèle indisponible (checkpoints ?)")
    if health["model_backend"] == "mock":
        st.sidebar.info("Mode démo : modèle simulé, les scores ne sont pas ceux de PatchCore.")
except requests.RequestException:
    st.sidebar.error(f"API injoignable sur {API_URL}. Lance : `uvicorn app.api.main:app`")
    st.stop()

category = st.sidebar.selectbox("Catégorie", CATEGORIES)
method = st.sidebar.selectbox("Méthode", list(METHOD_LABELS), format_func=METHOD_LABELS.get)

tab_analyse, tab_compare, tab_calib, tab_info = st.tabs(
    ["🖼️ Analyser", "💡 Comparer sous éclairage", "🎯 Recalibrer", "📊 État du système"])

# ------------------------------------------------------------------ 1. Analyser
with tab_analyse:
    st.subheader("Analyser une image")
    upload = st.file_uploader("Image de la pièce (PNG/JPG)", type=["png", "jpg", "jpeg"],
                              key="analyse")
    if upload and st.button("Analyser", type="primary"):
        with st.spinner("Analyse en cours…"):
            resp = api_predict(upload.getvalue(), upload.name, category, method)
        if resp.ok:
            col1, col2 = st.columns(2)
            col1.image(upload.getvalue(), caption="Image envoyée", width="stretch")
            with col2:
                show_result(resp.json())
        else:
            st.error(error_message(resp))

# ------------------------------------------------------------------ 2. Comparer
with tab_compare:
    st.subheader("Que se passe-t-il quand la lumière change ?")
    st.write("On analyse la même pièce sous l'éclairage d'origine, puis sous un éclairage modifié "
             "(mêmes transformations que les expériences du projet).")
    c1, c2 = st.columns(2)
    shift = c1.selectbox("Changement d'éclairage", list(SHIFT_LABELS), format_func=SHIFT_LABELS.get)
    severity = c2.select_slider("Sévérité", options=[1, 2, 3], value=2)
    upload_c = st.file_uploader("Image de la pièce", type=["png", "jpg", "jpeg"], key="compare")
    compare_methods = st.checkbox("Comparer aussi baseline vs LightShift sous l'éclairage modifié")

    if upload_c and st.button("Comparer", type="primary"):
        img = upload_c.getvalue()
        with st.spinner("Analyse…"):
            ref = api_predict(img, upload_c.name, category, method)
            shifted = api_predict(img, upload_c.name, category, method, shift, severity)
        if not (ref.ok and shifted.ok):
            st.error(error_message(ref if not ref.ok else shifted))
        else:
            left, right = st.columns(2)
            with left:
                st.image(img, caption="Éclairage d'origine", width="stretch")
                show_result(ref.json())
            with right:
                st.image(b64_image(shifted.json()["processed_image"]),
                         caption=f"{SHIFT_LABELS[shift]} · niveau {severity}", width="stretch")
                show_result(shifted.json())

            if compare_methods:
                st.divider()
                st.markdown("#### Sous l'éclairage modifié : baseline vs LightShift")
                cols = st.columns(2)
                for col, m in zip(cols, ["baseline", "lightshift"]):
                    with col:
                        r = api_predict(img, upload_c.name, category, m, shift, severity)
                        if r.ok:
                            show_result(r.json(), METHOD_LABELS[m])
                        else:
                            st.warning(f"{METHOD_LABELS[m]} : {error_message(r)}")

# ------------------------------------------------------------------ 3. Recalibrer
with tab_calib:
    st.subheader("Recalibrer le seuil pour un nouvel éclairage (brique B3)")
    st.write("Nouvelle lampe, nouvelle machine ? Envoie quelques photos de pièces **normales** prises "
             "sous ce nouvel éclairage : le seuil est recalculé pour limiter les fausses alertes.")
    alpha = st.select_slider("Taux de fausses alertes toléré (α)", options=[0.01, 0.05, 0.1, 0.2],
                             value=0.1, format_func=lambda a: f"{int(a * 100)} %")
    n_min = next(n for n in range(1, 1000) if math.ceil((n + 1) * (1 - alpha) - 1e-12) <= n)
    st.caption(f"Il faut au moins **{n_min} images normales** pour α = {int(alpha * 100)} %.")
    st.warning("N'utilise pas ensuite ces mêmes images pour évaluer le modèle (fuite de données).")

    calib = st.file_uploader("Images normales du nouvel éclairage", type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True, key="calib")
    col_a, col_b = st.columns(2)
    if col_a.button("Recalibrer", type="primary", disabled=not calib):
        files = [("images", (f.name, f.getvalue())) for f in calib]
        with st.spinner(f"Calcul des scores de {len(files)} images…"):
            resp = requests.post(f"{API_URL}/calibrate", files=files, timeout=600,
                                 data={"category": category, "alpha": str(alpha), "method": method})
        if resp.ok:
            r = resp.json()
            st.success(f"Nouveau seuil : **{r['new_threshold']:.3f}** (avant : {r['previous_threshold']:.3f}) "
                       f"— calculé sur {r['n_images']} images.")
            st.caption(r["note"])
        else:
            st.error(error_message(resp))
    if col_b.button("Revenir au seuil par défaut"):
        r = requests.delete(f"{API_URL}/calibrate/{category}", params={"method": method}, timeout=10)
        st.info(f"Seuil actuel : {r.json()['threshold']:.3f} ({r.json()['threshold_source']})")

# ------------------------------------------------------------------ 4. État
with tab_info:
    st.subheader("État du système")
    m = api_get("/metrics").json()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Analyses", m["predict_requests"])
    k2.metric("Défauts détectés", m["decisions"]["anomaly"])
    k3.metric("Recalibrations", m["calibrations"])
    k4.metric("Temps moyen", f"{m['mean_inference_ms']} ms" if m["mean_inference_ms"] else "—")
    st.markdown("**Seuils actuels**")
    st.table([{"méthode:catégorie": k, "seuil": v["threshold"], "source": v["source"]}
              for k, v in m["thresholds"].items()])
