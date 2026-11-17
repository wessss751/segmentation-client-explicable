"""Segmentation client explicable - Online Retail (UCI).

Modules :
    data           chargement, nettoyage, construction du profil client (RFM+)
    preprocessing  transformation log, valeurs extrêmes, standardisation
    pca            ACP normée et aides à l'interprétation
    clustering     K-Means, CAH de Ward, classification mixte, DBSCAN / HDBSCAN
    validation     indices internes, stabilité bootstrap, indices externes
    explain        profils, arbre de substitution, explication d'une affectation
    model          objet SegmentationModel (entraînement, prédiction, sauvegarde)
"""

__version__ = "1.0.0"
