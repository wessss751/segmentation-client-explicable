"""Module 2 (b) - ACP normée.

Les données sont déjà centrées-réduites : l'ACP sur la matrice des corrélations
(ACP normée) revient à une ACP sur Z. Les valeurs propres sont ramenées à la
convention « somme = nombre de variables » (règle de Kaiser : λ > 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Noms donnés aux axes après lecture du cercle des corrélations
# (vérifiés sur Online Retail ; recalculés automatiquement sinon)
AXIS_NAMES = {
    1: "Valeur et intensité d'achat (Montant, Fréquence, NbProduits)",
    2: "Ancienneté sans achat récent (Ancienneté, Récence)",
    3: "Gros paniers peu fréquents (PanierMoyen vs Fréquence)",
    4: "Retours (TauxRetour)",
}


@dataclass
class NormedPCA:
    n_axes: int | None = None  # None : choix automatique
    cum_threshold: float = 0.75

    features: list = field(default_factory=list)
    pca_: PCA | None = None
    eigenvalues_: np.ndarray | None = None
    rules_: dict = field(default_factory=dict)

    def fit(self, Z: pd.DataFrame) -> "NormedPCA":
        self.features = list(Z.columns)
        p = Z.shape[1]
        self.pca_ = PCA(n_components=p).fit(Z.to_numpy())
        ev = self.pca_.explained_variance_ratio_ * p
        self.eigenvalues_ = ev
        cum = np.cumsum(self.pca_.explained_variance_ratio_)
        kaiser = int((ev > 1).sum())
        # coude : plus forte cassure (différence seconde maximale) de l'éboulis
        if p >= 3:
            d2 = ev[:-2] - 2 * ev[1:-1] + ev[2:]
            elbow = int(np.argmax(d2) + 1)
        else:
            elbow = 1
        cum_rule = int(np.searchsorted(cum, self.cum_threshold) + 1)
        self.rules_ = {"kaiser": kaiser, "coude": elbow, "part_cumulee": cum_rule}
        if self.n_axes is None:
            # compromis : au moins Kaiser et au moins la part cumulée visée (2 axes minimum)
            self.n_axes = max(2, kaiser, cum_rule)
        return self

    # ------------------------------------------------------------------ #
    def transform(self, Z: pd.DataFrame, n: int | None = None) -> np.ndarray:
        n = n or self.n_axes
        return self.pca_.transform(Z[self.features].to_numpy())[:, :n]

    def eigen_table(self) -> pd.DataFrame:
        ratio = self.pca_.explained_variance_ratio_
        return pd.DataFrame({
            "Axe": [f"F{i+1}" for i in range(len(ratio))],
            "Valeur propre": self.eigenvalues_,
            "% inertie": 100 * ratio,
            "% cumulé": 100 * np.cumsum(ratio),
        })

    def correlations(self) -> pd.DataFrame:
        """Corrélation variable / axe = vecteur propre × sqrt(valeur propre)."""
        corr = self.pca_.components_.T * np.sqrt(self.eigenvalues_)
        return pd.DataFrame(corr, index=self.features, columns=[f"F{i+1}" for i in range(corr.shape[1])])

    def cos2(self) -> pd.DataFrame:
        c2 = self.correlations() ** 2
        return c2

    def contributions(self) -> pd.DataFrame:
        ctr = 100 * self.pca_.components_.T ** 2
        return pd.DataFrame(ctr, index=self.features, columns=[f"F{i+1}" for i in range(ctr.shape[1])])

    def auto_axis_name(self, axis: int, top: int = 3) -> str:
        """Nom proposé à partir des variables les plus corrélées à l'axe."""
        col = self.correlations()[f"F{axis}"]
        pos = col[col > 0.4].sort_values(ascending=False).index[:top].tolist()
        neg = col[col < -0.4].sort_values().index[:top].tolist()
        parts = []
        if pos:
            parts.append("+ " + ", ".join(pos))
        if neg:
            parts.append("− " + ", ".join(neg))
        return " / ".join(parts) if parts else "axe peu structuré"

    def axis_names(self) -> dict:
        return {i: AXIS_NAMES.get(i, self.auto_axis_name(i)) for i in range(1, self.n_axes + 1)}

    def individuals_cos2(self, Z: pd.DataFrame) -> np.ndarray:
        F = self.pca_.transform(Z[self.features].to_numpy())
        d2 = (F ** 2).sum(axis=1, keepdims=True)
        return F ** 2 / np.where(d2 == 0, 1, d2)
