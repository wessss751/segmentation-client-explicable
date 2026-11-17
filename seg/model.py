"""Objet SegmentationModel : chaîne complète entraînée, sauvegardée et utilisée
par l'application Streamlit et par l'API FastAPI.

    profil brut -> Preprocessor (log, standardisation, extrêmes) -> ACP normée
    -> partition (mixte / K-Means / CAH) -> centres -> affectation d'un nouveau
    client au centre le plus proche + explication.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from .clustering import assign_nearest, centers_of, fit_partition, r2_score_partition
from .data import FEATURES, ROOT
from .explain import (SHORT_LABELS, auto_names, describe_value, fit_surrogate_tree,
                      leaf_rule_for, profile_index, segment_profiles, tree_rules)
from .pca import NormedPCA
from .preprocessing import Preprocessor

MODEL_PATH = ROOT / "models" / "segmentation_model.joblib"


@dataclass
class SegmentationConfig:
    k: int = 5
    method: str = "mixte"
    outlier_mode: str = "ecreter"
    z_threshold: float = 3.0
    skew_threshold: float = 1.0
    n_axes: int | None = None
    tree_depth: int = 4
    atypical_quantile: float = 0.95
    seed: int = 0


@dataclass
class SegmentationModel:
    config: SegmentationConfig = field(default_factory=SegmentationConfig)

    preprocessor: Preprocessor | None = None
    pca: NormedPCA | None = None
    centers_: np.ndarray | None = None  # dans l'espace de l'algorithme (F ou Z)
    centers_z_: np.ndarray | None = None  # dans l'espace Z (pour expliquer par variable)
    labels_: np.ndarray | None = None
    distances_: np.ndarray | None = None
    dist_threshold_: dict = field(default_factory=dict)
    inliers_: np.ndarray | None = None
    names: dict = field(default_factory=dict)
    actions: dict = field(default_factory=dict)
    profiles_: pd.DataFrame | None = None
    index_: pd.DataFrame | None = None
    medians_: pd.DataFrame | None = None
    global_median_: pd.Series | None = None
    tree_: object = None
    tree_info_: dict = field(default_factory=dict)
    rules_: pd.DataFrame | None = None
    metrics_: dict = field(default_factory=dict)
    fit_info_: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def space(self) -> str:
        return "F" if self.config.method == "mixte" else "Z"

    def _to_space(self, Zdf: pd.DataFrame) -> np.ndarray:
        return self.pca.transform(Zdf) if self.space == "F" else Zdf[FEATURES].to_numpy()

    # ------------------------------------------------------------------ #
    def fit(self, prof: pd.DataFrame, meta: dict | None = None) -> "SegmentationModel":
        c = self.config
        X = prof[FEATURES]
        self.preprocessor = Preprocessor(skew_threshold=c.skew_threshold, outlier_mode=c.outlier_mode,
                                         z_threshold=c.z_threshold)
        Zdf = self.preprocessor.fit_transform(X)
        inl = ~self.preprocessor.outlier_mask_
        self.inliers_ = inl
        self.pca = NormedPCA(n_axes=c.n_axes).fit(Zdf[inl])
        F = self.pca.transform(Zdf)
        Z = Zdf.to_numpy()

        lab_in, centers, info = fit_partition(c.method, Z[inl], F[inl], c.k, c.seed)
        self.fit_info_ = {k: v for k, v in info.items() if k not in ("linkage", "labels_cah")}
        Xs = F if self.space == "F" else Z
        labels = np.empty(len(X), dtype=int)
        labels[inl] = lab_in
        if (~inl).any():
            labels[~inl] = assign_nearest(Xs[~inl], centers)[0]
        self.labels_ = labels
        self.centers_ = centers
        self.centers_z_ = centers_of(Z, labels, c.k)
        self.distances_ = np.linalg.norm(Xs - centers[labels], axis=1)
        self.dist_threshold_ = {
            int(s): float(np.quantile(self.distances_[labels == s], c.atypical_quantile)) for s in range(c.k)
        }

        self.names, self.actions = auto_names(prof, labels)
        self.global_median_ = X.median()
        self.medians_ = X.groupby(labels).median()
        self.profiles_ = segment_profiles(prof, labels, self.names)
        self.index_ = profile_index(prof, labels)
        self.tree_, self.tree_info_ = fit_surrogate_tree(prof, labels, c.tree_depth, c.seed)
        self.rules_ = tree_rules(self.tree_, self.names)
        self.metrics_ = {
            "silhouette": float(silhouette_score(Z, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
            "davies_bouldin": float(davies_bouldin_score(Z, labels)),
            "r2": r2_score_partition(Z, labels)[1],
        }
        self.meta = {"entraine_le": dt.datetime.now().isoformat(timespec="seconds"),
                     "n_clients": int(len(X)), **(meta or {})}
        return self

    def set_names(self, names: dict, actions: dict | None = None):
        self.names.update({int(k): v for k, v in names.items()})
        if actions:
            self.actions.update({int(k): v for k, v in actions.items()})
        self.profiles_["Nom"] = [self.names[s] for s in self.profiles_.index]
        self.rules_ = tree_rules(self.tree_, self.names)

    # ------------------------------------------------------------------ #
    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Zdf = self.preprocessor.transform(X[FEATURES])
        return assign_nearest(self._to_space(Zdf), self.centers_)

    def coords(self, X: pd.DataFrame) -> np.ndarray:
        """Coordonnées sur les axes ACP retenus (pour la carte)."""
        return self.pca.transform(self.preprocessor.transform(X[FEATURES]))

    def explain(self, x: dict | pd.Series, top: int = 3) -> dict:
        """Explication d'une affectation pour un client."""
        xs = pd.DataFrame([{f: float(x[f]) for f in FEATURES}])
        z_raw = self.preprocessor.standardize(xs).iloc[0]
        Zdf = self.preprocessor.transform(xs)
        Xs = self._to_space(Zdf)
        d_all = np.linalg.norm(self.centers_ - Xs[0], axis=1)
        order = np.argsort(d_all)
        seg = int(order[0])
        dist = float(d_all[seg])
        thr = self.dist_threshold_[seg]
        extreme_vars = [f for f in FEATURES if abs(z_raw[f]) > self.config.z_threshold]
        atypical = dist > thr or bool(extreme_vars)

        z = Zdf.iloc[0].to_numpy()
        gap_seg = z - self.centers_z_[seg]
        contrib = gap_seg ** 2 / max((gap_seg ** 2).sum(), 1e-12)
        med_seg = self.medians_.loc[seg]
        variables = []
        for j in np.argsort(-np.abs(z)):  # écart au profil moyen (client moyen : z = 0)
            f = FEATURES[j]
            variables.append({
                "variable": f,
                "libelle": SHORT_LABELS[f],
                "valeur": float(xs[f].iloc[0]),
                "z_score": float(z_raw[f]),
                "mediane_globale": float(self.global_median_[f]),
                "mediane_segment": float(med_seg[f]),
                "ecart_au_centre_segment": float(gap_seg[j]),
                "part_distance_segment": float(contrib[j]),
                "phrase": self._phrase(f, float(xs[f].iloc[0]), float(z_raw[f])),
            })
        reasons = []
        if dist > thr:
            reasons.append(f"distance au centre ({dist:.2f}) supérieure au {int(100*self.config.atypical_quantile)}e "
                           f"centile du segment ({thr:.2f})")
        if extreme_vars:
            reasons.append("valeur extrême sur : " + ", ".join(SHORT_LABELS[f] for f in extreme_vars))
        return {
            "segment": seg,
            "nom": self.names.get(seg, f"Segment {seg}"),
            "action": self.actions.get(seg, ""),
            "distance_centre": dist,
            "seuil_atypique": thr,
            "atypique": bool(atypical),
            "raisons_atypique": reasons,
            "segment_suivant": int(order[1]),
            "nom_segment_suivant": self.names.get(int(order[1]), ""),
            "distance_segment_suivant": float(d_all[order[1]]),
            "marge": float(d_all[order[1]] - dist),
            "distances": {int(s): float(d) for s, d in enumerate(d_all)},
            "variables_principales": variables[:top],
            "variables": variables,
            "regle_arbre": leaf_rule_for(self.tree_, self.rules_, xs[FEATURES].to_numpy()[0]),
            "segment_arbre": int(self.tree_.predict(xs[FEATURES].to_numpy())[0]),
        }

    def _phrase(self, f: str, v: float, z: float) -> str:
        med = float(self.global_median_[f])
        if abs(z) < 0.5:
            level = "proche de la moyenne"
        elif abs(z) < 1.5:
            level = "au-dessus de la moyenne" if z > 0 else "en dessous de la moyenne"
        else:
            level = "très au-dessus de la moyenne" if z > 0 else "très en dessous de la moyenne"
        return (f"{SHORT_LABELS[f]} = {describe_value(f, v)} : {level} "
                f"(médiane des clients {describe_value(f, med)}, z = {z:+.1f})")

    # ------------------------------------------------------------------ #
    def summary(self) -> list[dict]:
        out = []
        for s in range(self.config.k):
            p = self.profiles_.loc[s]
            out.append({
                "segment": s,
                "nom": self.names[s],
                "action": self.actions[s],
                "effectif": int(p["Effectif"]),
                "part_clients_pct": float(p["% clients"]),
                "part_ca_pct": float(p["Part du CA (%)"]),
                "medianes": {f: float(self.medians_.loc[s, f]) for f in FEATURES},
                "indices_profil": {f: float(self.index_.loc[s, f]) for f in FEATURES},
                "regles": self.rules_.loc[self.rules_["Segment"] == s, "Règle"].tolist(),
            })
        return out

    def save(self, path: str | Path = MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path = MODEL_PATH) -> "SegmentationModel":
        return joblib.load(path)

    def config_dict(self) -> dict:
        return asdict(self.config)
