
from anomalib.data import MVTecAD

# Chemin vers le dataset
datamodule = MVTecAD(
    root="data/raw/mvtec_anomaly_detection",
    category="metal_nut",
    train_batch_size=8,
    eval_batch_size=8,
    num_workers=0,
)

# Préparer les données
datamodule.setup()

# Afficher les informations
print("Dataset chargé avec succès !")
print("Catégorie : metal_nut")
print("Images d'entraînement :", len(datamodule.train_data))
print("Images de test :", len(datamodule.test_data))
