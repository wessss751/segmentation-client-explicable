"""Module 3 - Modélisation.

    K-Means (plusieurs initialisations)       -> sur les variables standardisées Z
    CAH de Ward + dendrogramme + cophénétique  -> sur Z
    Classification mixte                       -> CAH de Ward sur les axes retenus de
                                                  l'ACP, puis consolidation par K-Means
                                                  initialisé aux centres de la CAH
    DBSCAN / HDBSCAN                           -> sur Z (densité, détecte le bruit)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

METHODS = {
    "mixte": "Classification mixte (ACP → CAH Ward → K-Means)",
    "kmeans": "K-Means (variables standardisées)",
    "cah": "CAH de Ward (variables standardisées)",
}


# --------------------------------------------------------------------------- #
# Outils
# --------------------------------------------------------------------------- #
def centers_of(X: np.ndarray, labels: np.ndarray, k: int | None = None) -> np.ndarray:
    k = k or int(labels.max()) + 1
    return np.vstack([X[labels == j].mean(axis=0) for j in range(k)])


def assign_nearest(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    lab = d.argmin(axis=1)
    return lab, np.sqrt(d[np.arange(len(X)), lab])


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Numérote les segments du plus grand (0) au plus petit (bruit -1 conservé)."""
    valid = labels[labels >= 0]
    order = pd.Series(valid).value_counts().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping.get(l, -1) for l in labels])


def r2_score_partition(X: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(inertie intra, R² = inertie inter / inertie totale)."""
    mask = labels >= 0
    X, labels = X[mask], labels[mask]
    total = ((X - X.mean(axis=0)) ** 2).sum()
    within = sum(((X[labels == j] - X[labels == j].mean(axis=0)) ** 2).sum() for j in np.unique(labels))
    return float(within), float(1 - within / total)


# --------------------------------------------------------------------------- #
# Algorithmes
# --------------------------------------------------------------------------- #
def run_kmeans(X: np.ndarray, k: int, n_init: int = 20, seed: int = 0):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)
    return km.labels_, km.cluster_centers_, {"inertie": float(km.inertia_), "n_iter": int(km.n_iter_)}


def ward_linkage(X: np.ndarray) -> np.ndarray:
    return linkage(X, method="ward")


def cophenetic_corr(Lk: np.ndarray, X: np.ndarray) -> float:
    c, _ = cophenet(Lk, pdist(X))
    return float(c)


def cut_tree(Lk: np.ndarray, k: int) -> np.ndarray:
    return fcluster(Lk, t=k, criterion="maxclust") - 1


def run_cah(X: np.ndarray, k: int, Lk: np.ndarray | None = None):
    Lk = ward_linkage(X) if Lk is None else Lk
    lab = cut_tree(Lk, k)
    return lab, centers_of(X, lab, k), {"linkage": Lk}


def run_mixed(F: np.ndarray, k: int, Lk: np.ndarray | None = None, seed: int = 0):
    """Classification mixte : CAH de Ward sur les axes ACP puis consolidation K-Means."""
    Lk = ward_linkage(F) if Lk is None else Lk
    lab_cah = cut_tree(Lk, k)
    c0 = centers_of(F, lab_cah, k)
    km = KMeans(n_clusters=k, init=c0, n_init=1, random_state=seed).fit(F)
    moved = int((km.labels_ != lab_cah).sum())
    return km.labels_, km.cluster_centers_, {
        "linkage": Lk,
        "labels_cah": lab_cah,
        "deplaces": moved,
        "part_deplaces": moved / len(F),
        "inertie": float(km.inertia_),
    }


def fit_partition(method: str, Z: np.ndarray, F: np.ndarray, k: int, seed: int = 0, Lk=None):
    """Point d'entrée commun. Renvoie (labels triés par taille, centres dans l'espace
    de l'algorithme, infos). Espace : F (axes ACP) pour 'mixte', Z sinon."""
    if method == "mixte":
        lab, cen, info = run_mixed(F, k, Lk, seed)
        X = F
    elif method == "kmeans":
        lab, cen, info = run_kmeans(Z, k, seed=seed)
        X = Z
    elif method == "cah":
        lab, cen, info = run_cah(Z, k, Lk)
        X = Z
    else:
        raise ValueError(method)
    lab = relabel_by_size(lab)
    return lab, centers_of(X, lab, k), info


def run_dbscan(X: np.ndarray, eps: float, min_samples: int):
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    return relabel_by_size(lab)


def run_hdbscan(X: np.ndarray, min_cluster_size: int = 50, min_samples: int | None = None):
    lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(X)
    return relabel_by_size(lab)


# --------------------------------------------------------------------------- #
# Dendrogramme (coordonnées pour plotly)
# --------------------------------------------------------------------------- #
def dendrogram_coords(Lk: np.ndarray, p: int = 30) -> dict:
    return dendrogram(Lk, truncate_mode="lastp", p=p, no_plot=True, show_leaf_counts=True)


def cut_heights(Lk: np.ndarray, k_max: int = 10) -> pd.DataFrame:
    """Hauteurs de fusion des derniers regroupements (saut de Ward)."""
    h = Lk[::-1, 2][:k_max]
    return pd.DataFrame({"k (après coupure)": np.arange(2, len(h) + 2), "hauteur": h,
                         "saut": np.r_[h[:-1] - h[1:], np.nan]})


# --------------------------------------------------------------------------- #
# Comparaison des partitions
# --------------------------------------------------------------------------- #
def describe_partition(name: str, Z: np.ndarray, labels: np.ndarray, ref: np.ndarray | None = None,
                       extra: dict | None = None, sample: int = 4000, seed: int = 0) -> dict:
    mask = labels >= 0
    k = len(np.unique(labels[mask]))
    row = {"Méthode": name, "Nb segments": k, "% bruit": 100 * (1 - mask.mean())}
    if k >= 2:
        Xm, lm = Z[mask], labels[mask]
        row["Silhouette"] = silhouette_score(Xm, lm, sample_size=min(sample, len(Xm)), random_state=seed)
        row["Calinski-Harabasz"] = calinski_harabasz_score(Xm, lm)
        row["Davies-Bouldin"] = davies_bouldin_score(Xm, lm)
        row["R²"] = r2_score_partition(Z, labels)[1]
        sizes = np.bincount(lm)
        row["Plus petit segment (%)"] = 100 * sizes.min() / len(labels)
        row["Plus grand segment (%)"] = 100 * sizes.max() / len(labels)
    if ref is not None:
        row["ARI vs mixte"] = adjusted_rand_score(ref, labels)
    if extra:
        row.update(extra)
    return row
