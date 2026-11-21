
from anomalib.models import Patchcore

# Configuration du modèle PatchCore
model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
)

print("PatchCore configuré avec succès !")
print(model)
