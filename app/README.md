# Application LightShift-PatchCore (membre 3)

API FastAPI + interface Streamlit pour utiliser et démontrer PatchCore sous changement d'éclairage.

```
Image ──► /predict ──► modèle (mock ou PatchCore) ──► score + seuil ──► ✅ normal / ❌ défaut + carte
Images normales du nouvel éclairage ──► /calibrate ──► nouveau seuil (brique B3)
```

## Structure

```
app/
├── config.py                  # catégories, méthodes, variables d'environnement
├── api/
│   ├── main.py                # endpoints /health /predict /calibrate /metrics
│   └── schemas.py             # format des réponses (contrat D11)
├── core/
│   ├── conformal.py           # seuil conforme (B3)
│   ├── imaging.py             # lecture d'image, carte de chaleur, base64
│   ├── lighting.py            # réutilise apply_shift() du membre 1 pour la démo
│   └── thresholds.py          # sauvegarde des seuils recalibrés
├── models/
│   ├── base.py                # interface commune : predict() -> Prediction(score, anomaly_map)
│   ├── mock.py                # faux modèle (développement, tests)
│   ├── patchcore_backend.py   # vrais checkpoints anomalib (+ emplacement méthode membre 2)
│   └── registry.py            # choix du modèle via MODEL_BACKEND
└── ui/streamlit_app.py        # interface (4 onglets)
tests/                         # 34 tests pytest
docs/API_CONTRACT.md           # contrat d'API à valider en équipe
Dockerfile, docker-compose.yml
```

## Lancer en local (mode démo, sans checkpoints)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac : source .venv/bin/activate)
pip install -r requirements-app.txt

# Terminal 1 : l'API  ->  http://localhost:8000/docs
uvicorn app.api.main:app --reload

# Terminal 2 : l'interface  ->  http://localhost:8501
streamlit run app/ui/streamlit_app.py
```

## Passer au vrai PatchCore

1. Installer les dépendances lourdes : `pip install -r requirements.txt` (torch CPU + anomalib 2.7.1).
2. Placer les 3 checkpoints de la baseline ici :
   `results/Patchcore/MVTecAD/<category>/latest/weights/lightning/model.ckpt`
   (ou indiquer un autre dossier avec `CHECKPOINT_ROOT`).
3. Lancer l'API avec le vrai modèle :
   ```bash
   set MODEL_BACKEND=patchcore        # Linux/Mac : export MODEL_BACKEND=patchcore
   uvicorn app.api.main:app
   ```
4. Vérifier `/health` : `"status": "ok"` et `"model_backend": "patchcore"`.

⚠️ Le branchement PatchCore n'a pas encore pu être exécuté sans les checkpoints : c'est la
première chose à vérifier quand on les reçoit (voir la checklist plus bas).

## Docker

```bash
docker compose up --build                       # mode démo (léger)
# Vrai modèle : checkpoints dans results/Patchcore/ puis
INSTALL_PATCHCORE=true MODEL_BACKEND=patchcore docker compose up --build
```
API sur http://localhost:8000, interface sur http://localhost:8501.

## Tests

```bash
pytest -q
```

Couvre : contrat de sortie, images invalides/vides/manquantes, paramètres invalides,
déterminisme, calibration (mise à jour, minimum d'images, par catégorie, remise à zéro,
persistance), baisse des fausses alertes après recalibration, garantie FPR ≤ α sur données
simulées, modèle absent (503).

## Interface : les 4 onglets

| Onglet | Utilité |
|---|---|
| 🖼️ Analyser | Charger une image → décision, score, carte d'anomalie |
| 💡 Comparer sous éclairage | Même image sous l'éclairage d'origine vs modifié ; option baseline vs LightShift |
| 🎯 Recalibrer | Envoyer des images normales du nouvel éclairage → nouveau seuil (B3) |
| 📊 État du système | Nombre d'analyses, défauts, temps moyen, seuils actuels |

## Scénario de démo (D15)

1. Onglet Analyser : une vis **normale** (`test/good`) → ✅ ; une vis **défectueuse** → ❌ avec la zone en rouge.
2. Onglet Comparer : la vis normale sous `color_temperature` niveau 3 → le score monte, fausse alerte possible.
3. Onglet Recalibrer : envoyer ~20 images `train/good` transformées avec le même éclairage
   (dossier `data/processed/<category>/train/color_temperature/severity_3/good/`).
4. Retour à Comparer : même image → le seuil recalibré évite la fausse alerte, le défaut reste détecté.

## Checklist à la réception des checkpoints

- [ ] `/health` renvoie `ok` avec `MODEL_BACKEND=patchcore`
- [ ] `/predict` sur une image `test/good` et une image défectueuse : scores cohérents (défaut > normal)
- [ ] La carte d'anomalie est bien placée sur le défaut (comparer au masque `ground_truth`)
- [ ] Le score d'une image correspond à celui obtenu par les scripts d'évaluation du membre 1
- [ ] Noter le temps d'inférence CPU dans le rapport
