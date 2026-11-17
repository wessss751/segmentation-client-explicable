"""Module 4 - Choix du nombre de segments et validation.

Quatre angles :
    1. indices internes (k = 2..8) : inertie intra, R², silhouette, Calinski-Harabasz,
       Davies-Bouldin, + Dunn, gap statistic, BIC (mélange gaussien)
    2. stabilité par bootstrap : ARI moyen par k, Jaccard par segment
    3. indices externes : ARI, NMI, V-measure vs une étiquette jamais utilisée
    4. critères métier (taille minimale des segments, lisibilité) -> dans l'app
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             completeness_score, davies_bouldin_score,
                             homogeneity_score, normalized_mutual_info_score,
                             silhouette_samples, silhouette_score, v_measure_score)
from sklearn.mixture import GaussianMixture

from .clustering import (assign_nearest, centers_of, cut_tree, fit_partition,
                         r2_score_partition, ward_linkage)


# --------------------------------------------------------------------------- #
# Indices internes
# --------------------------------------------------------------------------- #
def dunn_index(X: np.ndarray, labels: np.ndarray, sample: int = 1500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    X, labels = X[idx], labels[idx]
    D = squareform(pdist(X))
    ks = np.unique(labels)
    diam = max(D[np.ix_(labels == a, labels == a)].max() for a in ks)
    sep = min(D[np.ix_(labels == a, labels == b)].min() for i, a in enumerate(ks) for b in ks[i + 1:])
    return float(sep / diam) if diam > 0 else np.nan


def gap_statistic(X: np.ndarray, labels: np.ndarray, k: int, B: int = 5, seed: int = 0) -> tuple[float, float]:
    """Gap(k) = E*[log W_k] - log W_k, référence uniforme dans la boîte englobante
    des données tournées sur leurs axes principaux (Tibshirani et al., 2001)."""
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xr = Xc @ Vt.T
    lo, hi = Xr.min(axis=0), Xr.max(axis=0)
    W = r2_score_partition(X, labels)[0]
    logs = []
    for _ in range(B):
        R = rng.uniform(lo, hi, size=Xr.shape) @ Vt
        km = KMeans(n_clusters=k, n_init=3, random_state=int(rng.integers(1e9))).fit(R)
        logs.append(np.log(km.inertia_))
    logs = np.array(logs)
    return float(logs.mean() - np.log(W)), float(logs.std() * np.sqrt(1 + 1 / B))


def internal_indices(method: str, Z: np.ndarray, F: np.ndarray, k_range=range(2, 9), seed: int = 0,
                     with_options: bool = True) -> pd.DataFrame:
    """Indices pour chaque k. Calculés dans l'espace Z (variables standardisées),
    commun à toutes les méthodes ; BIC et gap dans l'espace de l'algorithme."""
    X_fit = F if method == "mixte" else Z
    Lk = ward_linkage(X_fit) if method in ("mixte", "cah") else None
    rows = []
    for k in k_range:
        lab, _, _ = fit_partition(method, Z, F, k, seed, Lk)
        W, R2 = r2_score_partition(Z, lab)
        row = {
            "k": k,
            "Inertie intra": W,
            "R²": R2,
            "Silhouette": silhouette_score(Z, lab),
            "Calinski-Harabasz": calinski_harabasz_score(Z, lab),
            "Davies-Bouldin": davies_bouldin_score(Z, lab),
            "Taille min (%)": 100 * np.bincount(lab).min() / len(lab),
        }
        if with_options:
            row["Dunn"] = dunn_index(Z, lab, seed=seed)
            gap, s = gap_statistic(X_fit, lab, k, seed=seed)
            row["Gap"], row["Gap s.e."] = gap, s
            row["BIC (GMM)"] = GaussianMixture(k, covariance_type="full", n_init=2, random_state=seed).fit(X_fit).bic(X_fit)
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def best_k_by_index(tab: pd.DataFrame) -> dict:
    """k recommandé par chaque indice (règle de lecture de chaque indice)."""
    out = {
        "Silhouette (max)": int(tab["Silhouette"].idxmax()),
        "Calinski-Harabasz (max)": int(tab["Calinski-Harabasz"].idxmax()),
        "Davies-Bouldin (min)": int(tab["Davies-Bouldin"].idxmin()),
    }
    # coude de l'inertie : plus forte décroissance de la baisse relative
    W = tab["Inertie intra"].to_numpy()
    if len(W) >= 3:
        d2 = W[:-2] - 2 * W[1:-1] + W[2:]
        out["Coude de l'inertie"] = int(tab.index[np.argmax(d2) + 1])
    if "Dunn" in tab:
        out["Dunn (max)"] = int(tab["Dunn"].idxmax())
    if "Gap" in tab:
        ks = tab.index.to_list()
        gk = None
        for i in range(len(ks) - 1):  # plus petit k tel que Gap(k) >= Gap(k+1) - s(k+1)
            if tab["Gap"].iloc[i] >= tab["Gap"].iloc[i + 1] - tab["Gap s.e."].iloc[i + 1]:
                gk = ks[i]
                break
        out["Gap statistic (règle 1-s.e.)"] = int(gk or tab["Gap"].idxmax())
    if "BIC (GMM)" in tab:
        out["BIC GMM (min)"] = int(tab["BIC (GMM)"].idxmin())
    return out


def silhouette_detail(Z: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    s = silhouette_samples(Z, labels)
    df = pd.DataFrame({"segment": labels, "s": s})
    tab = df.groupby("segment")["s"].agg(
        Effectif="size", **{"Silhouette moyenne": "mean"}, **{"% silhouette négative": lambda x: 100 * (x < 0).mean()}
    )
    return s, tab


# --------------------------------------------------------------------------- #
# Stabilité par bootstrap
# --------------------------------------------------------------------------- #
def _boot_partition(method: str, Xb: np.ndarray, X_all: np.ndarray, k: int, seed: int, Lk=None):
    """Ajuste sur l'échantillon bootstrap, puis affecte tous les clients au centre le plus proche."""
    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Xb)
        return km.predict(X_all)
    lab_b = cut_tree(Lk, k)
    c = centers_of(Xb, lab_b, k)
    if method == "mixte":
        c = KMeans(n_clusters=k, init=c, n_init=1, random_state=seed).fit(Xb).cluster_centers_
    return assign_nearest(X_all, c)[0]


def jaccard_per_cluster(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    out = []
    for a in np.unique(ref):
        A = ref == a
        best = 0.0
        for b in np.unique(other):
            B = other == b
            inter = (A & B).sum()
            if inter:
                best = max(best, inter / (A | B).sum())
        out.append(best)
    return np.array(out)


def bootstrap_stability(method: str, Z: np.ndarray, F: np.ndarray, ref_labels: dict, B: int = 20,
                        seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """ref_labels : {k: étiquettes de référence sur toutes les données}.

    Renvoie (tableau ARI par k, {k: Jaccard moyen par segment})."""
    X = F if method == "mixte" else Z
    rng = np.random.default_rng(seed)
    ks = sorted(ref_labels)
    aris = {k: [] for k in ks}
    jacs = {k: [] for k in ks}
    for b in range(B):
        idx = rng.choice(len(X), len(X), replace=True)
        Xb = X[idx]
        Lk = ward_linkage(Xb) if method in ("mixte", "cah") else None
        for k in ks:
            lab = _boot_partition(method, Xb, X, k, seed + b, Lk)
            aris[k].append(adjusted_rand_score(ref_labels[k], lab))
            jacs[k].append(jaccard_per_cluster(ref_labels[k], lab))
    tab = pd.DataFrame({
        "k": ks,
        "ARI moyen": [np.mean(aris[k]) for k in ks],
        "ARI écart-type": [np.std(aris[k]) for k in ks],
        "ARI min": [np.min(aris[k]) for k in ks],
    }).set_index("k")
    jac = {k: np.mean(np.vstack(jacs[k]), axis=0) for k in ks}
    return tab, jac


def jaccard_reading(j: float) -> str:
    if j >= 0.85:
        return "très stable"
    if j >= 0.75:
        return "stable"
    if j >= 0.60:
        return "tendance réelle mais frontières floues"
    if j >= 0.50:
        return "instable"
    return "segment qui se dissout"


# --------------------------------------------------------------------------- #
# Indices externes
# --------------------------------------------------------------------------- #
def external_indices(labels: np.ndarray, target: np.ndarray) -> dict:
    return {
        "ARI": adjusted_rand_score(target, labels),
        "NMI": normalized_mutual_info_score(target, labels),
        "V-measure": v_measure_score(target, labels),
        "Homogénéité": homogeneity_score(target, labels),
        "Complétude": completeness_score(target, labels),
    }


def label_rate_by_segment(labels: np.ndarray, target: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"segment": labels, "y": np.asarray(target)})
    tab = df.groupby("segment")["y"].agg(Effectif="size", Taux="mean")
    tab["Taux"] *= 100
    tab["Écart au global (pts)"] = tab["Taux"] - 100 * df["y"].mean()
    tab["Lift"] = tab["Taux"] / (100 * df["y"].mean())
    return tab
