"""Module 2 (a) - Préparation des variables pour un calcul de distances.

Chaîne appliquée, dans cet ordre :
    1. log(1+x) sur les variables positives asymétriques (|skewness| > seuil)
    2. standardisation (centrage-réduction) de toutes les variables
    3. traitement explicite des valeurs extrêmes, au choix :
         - "conserver" : rien n'est modifié
         - "ecreter"   : z-scores écrêtés à ±seuil (winsorisation), tous les
                          clients restent dans le clustering
         - "a_part"    : clients avec un |z| > seuil suivis à part (exclus de
                          la construction des segments, affectés ensuite)
         - "dbscan"    : clients marqués « bruit » par DBSCAN isolés
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

OUTLIER_MODES = {
    "ecreter": "Écrêtées (winsorisation des z-scores)",
    "conserver": "Conservées telles quelles",
    "a_part": "Suivies à part (exclues de la construction des segments)",
    "dbscan": "Isolées par DBSCAN (points de bruit)",
}

OUTLIER_JUSTIFICATION = {
    "ecreter": (
        "Après passage au log, il reste quelques très gros comptes (grossistes) dont les "
        "z-scores dépassent 3 à 4. Les écrêter à ±{thr} garde tous les clients dans la "
        "segmentation tout en empêchant ces points d'attirer les centres du K-Means. "
        "Ils restent repérés par l'indicateur « client atypique »."
    ),
    "conserver": (
        "Toutes les valeurs sont gardées : la transformation log suffit à réduire "
        "l'asymétrie. Inconvénient : les K-Means et la CAH de Ward sont sensibles aux "
        "points extrêmes, qui peuvent former de petits segments à eux seuls."
    ),
    "a_part": (
        "Les clients ayant au moins une variable à plus de {thr} écarts-types sont des "
        "comptes exceptionnels (grossistes, très gros paniers, retours massifs). Ils sont "
        "suivis par un gestionnaire dédié et exclus de la construction des segments, puis "
        "affectés au segment le plus proche avec le drapeau « atypique »."
    ),
    "dbscan": (
        "DBSCAN repère les clients situés dans des zones de faible densité. Ces points de "
        "bruit sont isolés de la construction des segments puis affectés au segment le "
        "plus proche avec le drapeau « atypique »."
    ),
}


@dataclass
class Preprocessor:
    skew_threshold: float = 1.0
    outlier_mode: str = "ecreter"
    z_threshold: float = 3.0
    dbscan_eps: float | None = None
    dbscan_min_samples: int | None = None

    # appris au fit
    features: list = field(default_factory=list)
    log_vars: list = field(default_factory=list)
    skewness_: pd.DataFrame | None = None
    mean_: pd.Series | None = None
    std_: pd.Series | None = None
    outlier_mask_: np.ndarray | None = None  # True = valeur extrême (sur les données d'apprentissage)
    outlier_info_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.features = list(X.columns)
        sk_before = X.skew()
        self.log_vars = [c for c in self.features if X[c].min() >= 0 and abs(sk_before[c]) > self.skew_threshold]
        L = self._log(X)
        self.skewness_ = pd.DataFrame({
            "Asymétrie brute": sk_before,
            "Transformation": ["log(1+x)" if c in self.log_vars else "aucune" for c in self.features],
            "Asymétrie après": L.skew(),
        })
        self.mean_ = L.mean()
        self.std_ = L.std(ddof=0).replace(0, 1.0)
        Z_raw = (L - self.mean_) / self.std_
        self.outlier_mask_ = self._detect(Z_raw)
        return self

    def _log(self, X: pd.DataFrame) -> pd.DataFrame:
        L = X[self.features].astype(float).copy()
        for c in self.log_vars:
            L[c] = np.log1p(L[c].clip(lower=0))
        return L

    def standardize(self, X: pd.DataFrame) -> pd.DataFrame:
        """z-scores avant tout écrêtage."""
        return (self._log(X) - self.mean_) / self.std_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Données prêtes pour le calcul de distances."""
        Z = self.standardize(X)
        if self.outlier_mode == "ecreter":
            Z = Z.clip(-self.z_threshold, self.z_threshold)
        return Z

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------ #
    def _detect(self, Z: pd.DataFrame) -> np.ndarray:
        extreme = (Z.abs() > self.z_threshold).any(axis=1).to_numpy()
        info = {
            "mode": self.outlier_mode,
            "n_extremes_z": int(extreme.sum()),
            "part_extremes_z": float(extreme.mean()),
            "par_variable": (Z.abs() > self.z_threshold).sum().to_dict(),
        }
        if self.outlier_mode == "dbscan":
            ms = self.dbscan_min_samples or 2 * Z.shape[1]
            eps = self.dbscan_eps or suggest_eps(Z.to_numpy(), ms)
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z.to_numpy())
            mask = lab == -1
            info.update({"eps": float(eps), "min_samples": ms, "n_bruit": int(mask.sum())})
        elif self.outlier_mode == "a_part":
            mask = extreme
        else:
            mask = np.zeros(len(Z), dtype=bool)
        info["n_isoles"] = int(mask.sum())
        self.outlier_info_ = info
        return mask

    def is_extreme(self, X: pd.DataFrame) -> np.ndarray:
        """Indique si un client a une variable au-delà du seuil (avant écrêtage)."""
        return (self.standardize(X).abs() > self.z_threshold).any(axis=1).to_numpy()

    def justification(self) -> str:
        return OUTLIER_JUSTIFICATION[self.outlier_mode].format(thr=self.z_threshold)


def suggest_eps(X: np.ndarray, min_samples: int, quantile: float = 0.95) -> float:
    """eps de DBSCAN : quantile élevé de la distance au k-ième voisin (graphe k-distance)."""
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.quantile(d[:, -1], quantile))


def k_distance_curve(X: np.ndarray, min_samples: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, -1])
