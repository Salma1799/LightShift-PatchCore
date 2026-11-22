# Application LightShift-PatchCore (membre 3)

Une API FastAPI et une interface web « banc d'inspection » pour utiliser et démontrer PatchCore
sous changement d'éclairage. L'interface est servie par l'API elle-même : un seul programme à lancer.

```
Photo ──► /predict ──► modèle (démo ou PatchCore) ──► score + limite ──► Conforme / Défaut + zones en rouge
Pièces saines sous la nouvelle lumière ──► /calibrate ──► nouvelle limite (brique B3)
```

## Lancer en local

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/Mac : source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m uvicorn app.api.main:app --reload
```

Ouvrir ensuite :
- **http://localhost:8000** : l'interface
- **http://localhost:8000/docs** : la documentation de l'API

## L'interface

| Zone | À quoi elle sert |
|---|---|
| Cabine | Déposer une photo, ou essayer une pièce saine ou abîmée générée. Le curseur « Zones suspectes » règle l'affichage du rouge |
| Lumière de l'atelier | 6 lumières (normale, plus sombre, contraste faible, gamma, jaune, ombre) × 3 intensités. Ce sont les mêmes transformations que celles du membre 1 |
| Étiquette de verdict | « Conforme » ou « Défaut détecté », avec une phrase simple, une jauge score/limite et un conseil quand la lumière trompe le modèle |
| Méthode | Modèle d'origine ou LightShift pour le verdict principal |
| Les deux méthodes sur la même pièce | **Comparaison côte à côte** : même photo, même lumière, PatchCore d'origine à gauche, LightShift à droite |
| Comment le score dérive avec la lumière | **Vue de dérive** : score de la pièce sous les 15 lumières, pour les deux méthodes, avec la limite. Survol pour les valeurs, tableau dans « Voir les valeurs » |
| Recalibrer | Brique B3 : nouvelle limite calculée à partir de pièces saines sous la lumière choisie |
| Pièces inspectées | Historique des dernières inspections avec leur verdict |

## Le modèle de démonstration (`MODEL_BACKEND=mock`, par défaut)

C'est un **mini-PatchCore sans réseau de neurones** (`app/models/mock.py`). Il garde en mémoire des
patchs de pièces d'exemple saines et compare chaque patch au plus proche voisin. Il reproduit donc
le comportement du vrai modèle :

- une pièce abîmée est détectée ;
- un changement de lumière provoque de **fausses alertes** sur les pièces saines ;
- la **recalibration** (B3) supprime ces fausses alertes ;
- la méthode « LightShift » (normalisation par canal, idée de B2) résiste à la plupart des lumières,
  **sauf à l'ombre**, qui est un gradient spatial. C'est la même limite que dans les résultats du membre 1.

⚠️ Ses scores n'ont aucune valeur scientifique. Il est pensé pour les pièces d'exemple : une vraie
photo MVTec sera jugée « inconnue ». Pour MVTec, il faut le vrai modèle.

## Passer au vrai PatchCore

1. Installer : `python -m pip install -r requirements.txt` (torch CPU + anomalib 2.7.1).
2. Placer les 3 checkpoints : `results/Patchcore/MVTecAD/<category>/latest/weights/lightning/model.ckpt`.
3. Lancer :
   ```powershell
   $env:MODEL_BACKEND="patchcore"; python -m uvicorn app.api.main:app
   ```
4. Vérifier que l'en-tête de l'interface indique « PatchCore connecté ».

La méthode LightShift avec le vrai modèle attend le code du membre 2 :
`_predict_lightshift()` dans `app/models/patchcore_backend.py`.

### Checklist à la réception des checkpoints

- [ ] `/health` renvoie `ok` avec `MODEL_BACKEND=patchcore`
- [ ] Une image `test/good` et une image défectueuse donnent des scores cohérents (défaut > normal)
- [ ] Les zones rouges tombent sur le défaut (comparer au masque `ground_truth`)
- [ ] Le score d'une image correspond à celui des scripts d'évaluation du membre 1
- [ ] Noter le temps d'analyse CPU dans le rapport

## Intégration continue

`.github/workflows/tests.yml` lance à chaque push :
1. les tests sur Python 3.11 et 3.12 ;
2. la construction de l'image Docker, puis un appel à `/health` pour vérifier qu'elle démarre.

Le résultat apparaît dans l'onglet **Actions** du dépôt GitHub.

## Docker

```bash
docker compose up --build                                        # mode démo
INSTALL_PATCHCORE=true MODEL_BACKEND=patchcore docker compose up --build   # vrai modèle
```

## Tests

```bash
python -m pytest -q
```

Les 40 tests couvrent : le contrat de sortie, les images invalides, vides ou manquantes, les
paramètres invalides, le déterminisme, la calibration (mise à jour, minimum d'images, par
catégorie, remise à zéro, persistance), la baisse des fausses alertes après recalibration, la
robustesse de LightShift, la garantie FPR ≤ α sur données simulées, le modèle absent (503), les
pièces d'exemple, l'interface servie, la vue de dérive et les percentiles de latence.

## Scénario de démo pour le jury (≈ 2 minutes)

1. **Pièce saine, lumière normale** → Conforme.
2. **Pièce abîmée** → Défaut détecté, avec la rayure en rouge.
3. **Pièce saine + lumière jaune** → fausse alerte. Le conseil explique que c'est la lumière qui trompe le modèle.
4. **Recalibrer pour cette lumière** → la limite monte et la pièce saine redevient conforme.
5. **Pièce abîmée sous la même lumière** → toujours détectée.
6. **Les deux méthodes sur la même pièce** → sous lumière jaune, le modèle d'origine se trompe et LightShift non.
7. **Vue de dérive** → d'un coup d'œil, le score du modèle d'origine monte avec chaque lumière, alors que LightShift reste stable, **sauf à l'ombre**.

## Structure

```
app/
├── api/main.py            # endpoints + sert l'interface
├── api/schemas.py         # format des réponses (contrat D11)
├── core/conformal.py      # seuil conforme (B3)
├── core/imaging.py        # lecture d'image, zones rouges, base64
├── core/lighting.py       # réutilise apply_shift() du membre 1
├── core/thresholds.py     # sauvegarde des limites recalibrées
├── models/examples.py     # pièces d'exemple (écrou, vis, transistor)
├── models/mock.py         # mini-PatchCore de démonstration
├── models/patchcore_backend.py  # vrais checkpoints + emplacement du membre 2
├── models/registry.py     # choix du modèle (MODEL_BACKEND)
└── web/index.html         # interface (HTML/CSS/JS, aucune dépendance)
```
