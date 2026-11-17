"""Module 5 - Explicabilité des segments.

    - profils : effectif, médianes, indices de profil (moyenne segment / moyenne
      globale × 100), part du chiffre d'affaires
    - nom et action proposés pour chaque segment (règles lisibles, modifiables)
    - arbre de décision de substitution (profondeur 3-4), fidélité, règles en phrases
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from .data import FEATURE_LABELS, FEATURE_UNITS, FEATURES

SHORT_LABELS = {
    "Recence": "Récence",
    "Frequence": "Fréquence",
    "Montant": "Montant",
    "PanierMoyen": "Panier moyen",
    "NbProduits": "Nb produits",
    "Anciennete": "Ancienneté",
    "TauxRetour": "Taux de retour",
}
INTEGER_VARS = {"Recence", "Frequence", "NbProduits", "Anciennete"}

# Actions associées aux noms de segments
ACTIONS = {
    "Champions": "Programme VIP : gestionnaire de compte dédié, accès en avant-première aux "
                 "nouvelles collections, livraison offerte. Pas de remise systématique : ces clients "
                 "achètent déjà, on protège la marge et la relation.",
    "Clients réguliers": "Montée en gamme : recommandations croisées (produits souvent achetés "
                         "ensemble), programme de fidélité à paliers pour les amener au niveau Champions.",
    "Nouveaux clients à gros panier": "Contact commercial sous 15 jours : ce sont souvent des revendeurs "
                                      "qui testent le site. Proposer un tarif volume / compte professionnel "
                                      "pour transformer le premier achat en relation régulière.",
    "Nouveaux petits acheteurs": "Parcours de bienvenue : e-mails d'onboarding, bon de réduction valable "
                                 "sur la 2e commande dans les 30 jours. L'objectif est le deuxième achat.",
    "Nouveaux clients": "Parcours de bienvenue et relance personnalisée après 30 jours pour déclencher "
                        "le deuxième achat.",
    "Clients endormis": "Campagne de réactivation peu coûteuse (e-mail « vous nous manquez » + offre "
                        "limitée dans le temps). Arrêter les envois aux clients qui ne réagissent pas.",
    "Anciens bons clients à reconquérir": "Appel ou e-mail personnalisé pour comprendre le départ, offre "
                                          "de reconquête ciblée : ce sont des clients rentables à risque.",
    "Acheteurs occasionnels à gros panier": "Relances saisonnières (avant Noël, rentrée) avec offres sur "
                                            "volume : ils achètent rarement mais beaucoup.",
    "Petits clients occasionnels": "Communication de masse à faible coût (newsletter), pas d'effort "
                                   "commercial individuel.",
    "Clients occasionnels": "Newsletter et offres saisonnières ; suivre leur évolution vers les "
                            "segments réguliers.",
}


# --------------------------------------------------------------------------- #
# Profils
# --------------------------------------------------------------------------- #
def segment_profiles(prof: pd.DataFrame, labels: np.ndarray, names: dict | None = None) -> pd.DataFrame:
    df = prof[FEATURES].copy()
    df["segment"] = labels
    g = df.groupby("segment")
    out = g[FEATURES].median().add_prefix("Médiane ")
    out.insert(0, "% clients", 100 * g.size() / len(df))
    out.insert(0, "Effectif", g.size())
    out["Part du CA (%)"] = 100 * g["Montant"].sum() / df["Montant"].sum()
    if names:
        out.insert(0, "Nom", [names.get(s, f"Segment {s}") for s in out.index])
    return out


def profile_index(prof: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Indice de profil = moyenne du segment / moyenne globale × 100 (100 = moyenne)."""
    df = prof[FEATURES].copy()
    means = df.groupby(labels).mean()
    return 100 * means / df.mean()


# --------------------------------------------------------------------------- #
# Noms et actions
# --------------------------------------------------------------------------- #
def auto_names(prof: pd.DataFrame, labels: np.ndarray) -> tuple[dict, dict]:
    """Nom et action par segment, déduits des médianes relatives (segment / global)."""
    med = prof[FEATURES].median()
    seg = prof[FEATURES].groupby(labels).median()
    rel = seg / med.replace(0, np.nan)
    names = {}
    for s, row in rel.iterrows():
        r, f, m, a, p = row["Recence"], row["Frequence"], row["Montant"], row["Anciennete"], row["PanierMoyen"]
        if r >= 1.8:
            name = "Anciens bons clients à reconquérir" if m >= 1.5 else "Clients endormis"
        elif f >= 2 and m >= 2:
            name = "Champions"
        elif a <= 0.7:
            if p >= 1.5 or m >= 1.3:
                name = "Nouveaux clients à gros panier"
            elif m <= 0.6:
                name = "Nouveaux petits acheteurs"
            else:
                name = "Nouveaux clients"
        elif f >= 1.2:
            name = "Clients réguliers"
        elif p >= 1.5:
            name = "Acheteurs occasionnels à gros panier"
        elif m <= 0.6:
            name = "Petits clients occasionnels"
        else:
            name = "Clients occasionnels"
        names[int(s)] = name
    # dédoublonnage
    counts = pd.Series(names).value_counts()
    seen = {}
    for s in sorted(names):
        n = names[s]
        if counts[n] > 1:
            seen[n] = seen.get(n, 0) + 1
            names[s] = f"{n} ({seen[n]})"
    actions = {s: ACTIONS.get(n.split(" (")[0], ACTIONS["Clients occasionnels"]) for s, n in names.items()}
    return names, actions


# --------------------------------------------------------------------------- #
# Arbre de substitution
# --------------------------------------------------------------------------- #
def fit_surrogate_tree(prof: pd.DataFrame, labels: np.ndarray, depth: int = 3, seed: int = 0):
    X = prof[FEATURES].to_numpy()
    leaf = max(20, int(0.01 * len(X)))
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed).fit(X, labels)
    fid = float((tree.predict(X) == labels).mean())
    cv = float(cross_val_score(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed),
                               X, labels, cv=5).mean())
    return tree, {"fidelite": fid, "fidelite_cv": cv, "profondeur": depth, "feuilles": int(tree.get_n_leaves())}


def _fmt_num(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:,.0f}".replace(",", " ")
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.1f}".replace(".", ",")


def _condition_text(var: str, lo: float | None, hi: float | None) -> str:
    """lo < x <= hi, traduit en français avec unités."""
    unit = FEATURE_UNITS.get(var, "")
    lab = SHORT_LABELS[var]
    if var in INTEGER_VARS:
        lo_i = int(np.floor(lo)) + 1 if lo is not None else None
        hi_i = int(np.floor(hi)) if hi is not None else None
        if lo_i is not None and hi_i is not None:
            return f"{lab} entre {lo_i} et {hi_i} {unit}" if lo_i != hi_i else f"{lab} = {lo_i} {unit}"
        if lo_i is not None:
            return f"{lab} ≥ {lo_i} {unit}"
        return f"{lab} ≤ {hi_i} {unit}"
    u = f" {unit}" if unit else ""
    if lo is not None and hi is not None:
        return f"{lab} entre {_fmt_num(lo)} et {_fmt_num(hi)}{u}"
    if lo is not None:
        return f"{lab} > {_fmt_num(lo)}{u}"
    return f"{lab} ≤ {_fmt_num(hi)}{u}"


def tree_rules(tree: DecisionTreeClassifier, names: dict | None = None) -> pd.DataFrame:
    """Une ligne par feuille : conditions fusionnées par variable, segment prédit, pureté."""
    t = tree.tree_
    classes = tree.classes_
    rows = []

    def walk(node, bounds):
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            n = int(t.n_node_samples[node])
            cls = int(classes[int(np.argmax(counts))])
            purity = float(counts.max() / counts.sum())
            conds = [_condition_text(v, lo, hi) for v, (lo, hi) in bounds.items()]
            seg_name = names.get(cls, f"Segment {cls}") if names else f"Segment {cls}"
            rows.append({
                "Segment": cls,
                "Nom": seg_name,
                "Clients": n,
                "Pureté (%)": 100 * purity,
                "Règle": "Si " + " et ".join(conds) + f", alors le client est classé « {seg_name} »"
                         f" ({n} clients, {100 * purity:.0f} % bien classés).",
            })
            return
        var = FEATURES[t.feature[node]]
        thr = float(t.threshold[node])
        lo, hi = bounds.get(var, (None, None))
        left = dict(bounds)
        left[var] = (lo, thr if hi is None else min(hi, thr))
        walk(t.children_left[node], left)
        right = dict(bounds)
        right[var] = (thr if lo is None else max(lo, thr), hi)
        walk(t.children_right[node], right)

    walk(0, {})
    return pd.DataFrame(rows).sort_values(["Segment", "Clients"], ascending=[True, False]).reset_index(drop=True)


def leaf_rule_for(tree: DecisionTreeClassifier, rules: pd.DataFrame, x: np.ndarray) -> str:
    """Règle de la feuille atteinte par un client (même ordre que tree_rules)."""
    leaf = tree.apply(x.reshape(1, -1))[0]
    t = tree.tree_
    path = []

    def collect(node, bounds):
        if t.children_left[node] == -1:
            if node == leaf:
                path.append(bounds)
            return
        var = FEATURES[t.feature[node]]
        thr = float(t.threshold[node])
        lo, hi = bounds.get(var, (None, None))
        collect(t.children_left[node], {**bounds, var: (lo, thr if hi is None else min(hi, thr))})
        collect(t.children_right[node], {**bounds, var: (thr if lo is None else max(lo, thr), hi)})

    collect(0, {})
    if not path:
        return ""
    return " et ".join(_condition_text(v, lo, hi) for v, (lo, hi) in path[0].items())


def describe_value(var: str, v: float) -> str:
    unit = FEATURE_UNITS.get(var, "")
    return f"{_fmt_num(v)} {unit}".strip()


def feature_label(var: str) -> str:
    return FEATURE_LABELS.get(var, var)
