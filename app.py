"""Application web interactive de segmentation explicable - Online Retail (UCI).

Lancement :  streamlit run app.py
"""
from __future__ import annotations

import copy
import io
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from seg.clustering import (METHODS, cophenetic_corr, cut_heights, dendrogram_coords,
                            describe_partition, fit_partition, run_dbscan, run_hdbscan,
                            ward_linkage)
from seg.data import (FEATURE_LABELS, FEATURES, ProfileConfig, build_customer_profile,
                      clean_transactions, load_transactions)
from seg.explain import SHORT_LABELS
from seg.model import MODEL_PATH, SegmentationConfig, SegmentationModel
from seg.preprocessing import OUTLIER_MODES, k_distance_curve, suggest_eps
from seg.validation import (best_k_by_index, bootstrap_stability, external_indices,
                            internal_indices, jaccard_reading, label_rate_by_segment,
                            silhouette_detail)

st.set_page_config(page_title="Segmentation explicable", page_icon="🧭", layout="wide")

# Palette catégorielle (ordre fixe, une couleur par segment) et divergente
SEG_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NOISE_COLOR = "#a3a29c"
DIVERGING = [[0, "#2a78d6"], [0.5, "#f1f0ed"], [1, "#e34948"]]
K_RANGE = range(2, 9)
TEMPLATE = "plotly_white"
HL = "background-color: rgba(42,120,214,0.35); font-weight: 600"  # surlignage lisible en clair et en sombre


def seg_color(s: int) -> str:
    return NOISE_COLOR if s < 0 else SEG_COLORS[s % len(SEG_COLORS)]


def fmt(v, d=2):
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


# --------------------------------------------------------------------------- #
# Calculs mis en cache
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Chargement et nettoyage des transactions…")
def get_transactions(file_bytes: bytes | None, filename: str | None):
    src = io.BytesIO(file_bytes) if file_bytes is not None else None
    tx = load_transactions(src, filename)
    tx, report = clean_transactions(tx)
    return tx, report


@st.cache_data(show_spinner="Construction du profil client…")
def get_profile(_tx, tx_key, use_label: bool, horizon: int):
    return build_customer_profile(_tx, ProfileConfig(use_label=use_label, horizon_days=horizon))


@st.cache_data(show_spinner="Entraînement du modèle…")
def get_model(prof: pd.DataFrame, cfg: tuple) -> SegmentationModel:
    return SegmentationModel(SegmentationConfig(*cfg)).fit(prof)


def spaces(model: SegmentationModel, prof: pd.DataFrame):
    """Z complet, F complet, et masque des clients utilisés pour construire les segments."""
    Zdf = model.preprocessor.transform(prof[FEATURES])
    F = model.pca.transform(Zdf)
    return Zdf, F, model.inliers_


@st.cache_data(show_spinner="Calcul des indices internes pour k = 2 à 8…")
def get_indices(prof, prep_cfg: tuple, method: str, with_options: bool):
    m = get_model(prof, prep_cfg)
    Zdf, F, inl = spaces(m, prof)
    return internal_indices(method, Zdf.to_numpy()[inl], F[inl], K_RANGE, with_options=with_options)


@st.cache_data(show_spinner="Stabilité par bootstrap (peut prendre ~30 s)…")
def get_stability(prof, prep_cfg: tuple, method: str, B: int):
    m = get_model(prof, prep_cfg)
    Zdf, F, inl = spaces(m, prof)
    Z, Fi = Zdf.to_numpy()[inl], F[inl]
    Lk = ward_linkage(Fi if method == "mixte" else Z) if method != "kmeans" else None
    refs = {k: fit_partition(method, Z, Fi, k, 0, Lk)[0] for k in K_RANGE}
    tab, jac = bootstrap_stability(method, Z, Fi, refs, B=B)
    return tab, jac, refs


@st.cache_data(show_spinner="CAH de Ward et corrélation cophénétique…")
def get_linkages(prof, prep_cfg: tuple):
    m = get_model(prof, prep_cfg)
    Zdf, F, inl = spaces(m, prof)
    Z, Fi = Zdf.to_numpy()[inl], F[inl]
    Lz, Lf = ward_linkage(Z), ward_linkage(Fi)
    return Lz, Lf, cophenetic_corr(Lz, Z), cophenetic_corr(Lf, Fi)


@st.cache_data(show_spinner="Comparaison des algorithmes…")
def get_comparison(prof, prep_cfg: tuple, k: int, eps: float, min_samples: int, min_cluster: int):
    m = get_model(prof, prep_cfg)
    Zdf, F, inl = spaces(m, prof)
    Z, Fi = Zdf.to_numpy()[inl], F[inl]
    parts, extras = {}, {}
    for meth in ["mixte", "kmeans", "cah"]:
        lab, _, info = fit_partition(meth, Z, Fi, k)
        parts[METHODS[meth]] = lab
        if meth == "mixte":
            extras[METHODS[meth]] = {"Remarque": f"{info['deplaces']} clients déplacés à la consolidation"}
    parts[f"DBSCAN (eps={eps:.2f}, minPts={min_samples})"] = run_dbscan(Z, eps, min_samples)
    parts[f"HDBSCAN (taille min {min_cluster})"] = run_hdbscan(Z, min_cluster)
    ref = parts[METHODS["mixte"]]
    rows = [describe_partition(n, Z, lab, ref, extras.get(n)) for n, lab in parts.items()]
    return pd.DataFrame(rows).set_index("Méthode"), parts


# --------------------------------------------------------------------------- #
# Barre latérale : paramètres
# --------------------------------------------------------------------------- #
st.sidebar.title("🧭 Segmentation explicable")
st.sidebar.caption("Online Retail (UCI) · RFM + indicateurs · ACP · clustering · explicabilité")

with st.sidebar.expander("1 · Données", expanded=True):
    source = st.radio("Source", ["Téléchargement automatique (UCI)", "Importer un fichier"], label_visibility="collapsed")
    up = None
    if source == "Importer un fichier":
        up = st.file_uploader("Fichier de factures (.xlsx / .csv)", type=["xlsx", "csv"],
                              help="Colonnes : InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country")
    use_label = st.checkbox("Construire l'étiquette externe « Réachat »", value=True,
                            help="Le profil est calculé avant une date de coupure ; l'étiquette vaut 1 si le client "
                                 "rachète dans la fenêtre suivante. Elle n'entre jamais dans le clustering.")
    horizon = st.slider("Fenêtre de réachat (jours)", 30, 120, 90, 15, disabled=not use_label)

with st.sidebar.expander("2 · Préparation", expanded=True):
    skew_thr = st.slider("Seuil d'asymétrie pour le log", 0.5, 2.0, 1.0, 0.25)
    outlier_mode = st.selectbox("Valeurs extrêmes", list(OUTLIER_MODES), format_func=OUTLIER_MODES.get)
    z_thr = st.slider("Seuil |z| d'une valeur extrême", 2.5, 5.0, 3.0, 0.5)
    axes_auto = st.checkbox("Nombre d'axes ACP automatique", value=True)
    n_axes = None if axes_auto else st.slider("Axes ACP retenus", 2, 6, 3)

with st.sidebar.expander("3 · Segmentation", expanded=True):
    method = st.selectbox("Méthode retenue", list(METHODS), format_func=METHODS.get)
    k = st.slider("Nombre de segments k", 2, 8, 5)
    depth = st.radio("Profondeur de l'arbre de substitution", [3, 4], index=1, horizontal=True)
    B = st.slider("Réplications bootstrap", 5, 50, 20, 5)
    with_options = st.checkbox("Indices optionnels (Dunn, gap, BIC)", value=True)

# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
try:
    tx, clean_report = get_transactions(up.getvalue() if up is not None else None, up.name if up is not None else None)
except Exception as e:  # noqa: BLE001
    st.error(f"Impossible de charger les données : {e}")
    st.stop()

if source == "Importer un fichier" and up is None:
    st.info("Aucun fichier importé : utilisation du jeu UCI.")

tx_key = (up.name, len(up.getvalue())) if up is not None else "uci"
prof, prof_info = get_profile(tx, tx_key, use_label, horizon)
has_label = "Reachat" in prof

cfg = (k, method, outlier_mode, z_thr, skew_thr, n_axes, depth)
prep_cfg = (5, "mixte", outlier_mode, z_thr, skew_thr, n_axes, 3)  # préparation commune à tous les k
model = copy.deepcopy(get_model(prof, cfg))

# Noms et actions modifiables (gardés en session pour la configuration courante)
sig = f"{cfg}-{tx_key}-{use_label}-{horizon}"
store = st.session_state.setdefault("names", {}).setdefault(sig, {})
if store:
    model.set_names({s: v["nom"] for s, v in store.items()}, {s: v["action"] for s, v in store.items()})

Zdf, F_all, inliers = spaces(model, prof)
Z_all = Zdf.to_numpy()
labels = model.labels_
names = model.names

st.title("Application web de segmentation client explicable")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clients segmentés", f"{len(prof):,}".replace(",", " "))
c2.metric("Segments (k)", k)
c3.metric("Silhouette", fmt(model.metrics_["silhouette"], 3))
c4.metric("R² (inertie expliquée)", f"{100 * model.metrics_['r2']:.1f} %")
c5.metric("Fidélité de l'arbre", f"{100 * model.tree_info_['fidelite']:.1f} %")

tabs = st.tabs(["1 · Données & profil", "2 · Préparation & ACP", "3 · Modélisation",
                "4 · Choix de k & validation", "5 · Segments expliqués", "6 · Nouveau client & API"])

# =========================================================================== #
# 1. Données et profil client
# =========================================================================== #
with tabs[0]:
    st.header("Module 1 – Données et profil client")
    a, b, c, d = st.columns(4)
    a.metric("Lignes de factures nettoyées", f"{len(tx):,}".replace(",", " "))
    p0, p1 = prof_info["periode_donnees"]
    b.metric("Période des données", f"{p0:%d/%m/%Y} → {p1:%d/%m/%Y}")
    c.metric("Fin de la fenêtre d'observation", f"{prof_info['fin_observation']:%d/%m/%Y}")
    d.metric("Taux de réachat (étiquette externe)",
             f"{100 * prof_info['taux_reachat']:.1f} %" if has_label else "—")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Nettoyage des transactions")
        st.dataframe(clean_report, hide_index=True, width="stretch")
        st.caption("Les annulations (factures « C… ») sont gardées pour calculer le taux de retour, "
                   f"puis {prof_info['n_clients_montant_negatif_retires']} clients au montant net nul ou négatif sont retirés.")
    with col2:
        st.subheader("Variables du profil client")
        st.dataframe(pd.DataFrame({"Variable": FEATURES, "Définition": [FEATURE_LABELS[f] for f in FEATURES],
                                   "Bloc": ["RFM", "RFM", "RFM", "Ajout", "Ajout", "Ajout", "Ajout"]}),
                     hide_index=True, width="stretch")
        if has_label:
            st.info(f"**Variables exclues du clustering** : `Reachat` (achat entre le {prof_info['fin_observation']:%d/%m/%Y} "
                    f"et le {prof_info['fin_horizon']:%d/%m/%Y}), `Pays`, `RoyaumeUni`. Elles servent uniquement à "
                    "évaluer les segments (indices externes, module 4).")

    st.subheader("Table clients")
    st.dataframe(prof.head(200), width="stretch", height=230)
    st.dataframe(prof[FEATURES].describe().T.style.format("{:,.1f}"), width="stretch")

    st.subheader("Distributions")
    var = st.selectbox("Variable", FEATURES, format_func=lambda f: FEATURE_LABELS[f], key="dist_var")
    h1, h2 = st.columns(2)
    fig = px.histogram(prof, x=var, nbins=60, template=TEMPLATE, color_discrete_sequence=[SEG_COLORS[0]],
                       title=f"{SHORT_LABELS[var]} – échelle brute")
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    h1.plotly_chart(fig, width="stretch")
    fig = px.histogram(x=np.log1p(prof[var]), nbins=60, template=TEMPLATE, color_discrete_sequence=[SEG_COLORS[0]],
                       title=f"{SHORT_LABELS[var]} – après log(1+x)", labels={"x": f"log(1+{var})"})
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    h2.plotly_chart(fig, width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("Corrélations (Spearman)")
        corr = prof[FEATURES].corr(method="spearman")
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=DIVERGING, template=TEMPLATE)
        g1.plotly_chart(fig, width="stretch")
    with g2:
        st.subheader("Valeurs extrêmes (z-scores après log)")
        zlong = model.preprocessor.standardize(prof[FEATURES]).melt(var_name="Variable", value_name="z")
        fig = px.box(zlong, x="Variable", y="z", template=TEMPLATE, color_discrete_sequence=[SEG_COLORS[0]])
        fig.add_hline(y=z_thr, line_dash="dot", line_color="#e34948")
        fig.add_hline(y=-z_thr, line_dash="dot", line_color="#e34948")
        g2.plotly_chart(fig, width="stretch")

    top_c = prof["Pays"].value_counts().head(10).sort_values()
    fig = px.bar(x=top_c.values, y=top_c.index, orientation="h", template=TEMPLATE, height=320,
                 color_discrete_sequence=[SEG_COLORS[0]], labels={"x": "Clients", "y": ""},
                 title="Pays des clients (10 premiers) – variable descriptive, hors clustering")
    st.plotly_chart(fig, width="stretch")

# =========================================================================== #
# 2. Préparation et ACP
# =========================================================================== #
with tabs[1]:
    st.header("Module 2 – Préparation et ACP normée")
    pp = model.preprocessor
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Transformation log puis standardisation")
        st.dataframe(pp.skewness_.style.format({"Asymétrie brute": "{:.2f}", "Asymétrie après": "{:.2f}"}),
                     width="stretch")
        st.caption(f"Règle : log(1+x) pour les variables positives dont |asymétrie| > {skew_thr}, appliqué "
                   "avant le centrage-réduction de toutes les variables. Récence et Ancienneté, bornées par la "
                   "période observée, restent sur leur échelle.")
    with c2:
        st.subheader("Traitement des valeurs extrêmes")
        info = pp.outlier_info_
        st.markdown(f"**Choix : {OUTLIER_MODES[outlier_mode]}**")
        st.write(pp.justification())
        st.markdown(f"- Clients avec au moins un |z| > {z_thr} : **{info['n_extremes_z']}** "
                    f"({100 * info['part_extremes_z']:.1f} %)\n"
                    f"- Clients isolés de la construction des segments : **{info['n_isoles']}**"
                    + (f"\n- DBSCAN : eps = {info['eps']:.2f}, minPts = {info['min_samples']}" if "eps" in info else ""))
        pv = pd.Series(info["par_variable"]).rename("Clients au-delà du seuil")
        st.dataframe(pv.to_frame().T, width="stretch")

    st.subheader("Valeurs propres et nombre d'axes")
    pca = model.pca
    eig = pca.eigen_table()
    e1, e2 = st.columns([1, 1.2])
    e1.dataframe(eig.style.format({"Valeur propre": "{:.3f}", "% inertie": "{:.1f}", "% cumulé": "{:.1f}"}),
                 hide_index=True, width="stretch")
    r = pca.rules_
    e1.markdown(f"- **Kaiser** (λ > 1) : {r['kaiser']} axes\n- **Coude** de l'éboulis : {r['coude']} axe(s)\n"
                f"- **Part cumulée ≥ {int(100 * pca.cum_threshold)} %** : {r['part_cumulee']} axes\n\n"
                f"➡️ **{pca.n_axes} axes retenus** ({eig['% cumulé'].iloc[pca.n_axes - 1]:.1f} % de l'inertie).")
    if pca.n_axes > r["kaiser"]:
        e1.caption(f"Kaiser strict donne {r['kaiser']} axes, mais λ{r['kaiser'] + 1} = "
                   f"{pca.eigenvalues_[r['kaiser']]:.2f} est très proche de 1 et l'axe a un sens clair : on le garde "
                   "pour atteindre la part cumulée visée.")
    fig = go.Figure(go.Bar(x=eig["Axe"], y=eig["Valeur propre"],
                           marker_color=[SEG_COLORS[0] if i < pca.n_axes else "#c3c2b7" for i in range(len(eig))],
                           text=eig["% inertie"].map(lambda v: f"{v:.1f} %"), textposition="outside"))
    fig.add_hline(y=1, line_dash="dot", annotation_text="Kaiser λ = 1", line_color="#52514e")
    fig.update_layout(template=TEMPLATE, title="Éboulis des valeurs propres (en bleu : axes retenus)",
                      yaxis_title="Valeur propre", height=380)
    e2.plotly_chart(fig, width="stretch")

    st.subheader("Cercle des corrélations et nom des axes")
    corr_v = pca.correlations()
    n_show = max(pca.n_axes, 2)
    pairs = [f"F{i} × F{j}" for i in range(1, n_show + 1) for j in range(i + 1, n_show + 1)]
    pair = [int(v.strip()[1:]) for v in st.selectbox("Plan factoriel", pairs).split("×")]
    cc1, cc2 = st.columns([1, 1])
    fx, fy = f"F{pair[0]}", f"F{pair[1]}"
    fig = go.Figure()
    t = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=np.cos(t), y=np.sin(t), mode="lines", line=dict(color="#c3c2b7", width=1),
                             hoverinfo="skip", showlegend=False))
    for f in FEATURES:
        x, y = corr_v.loc[f, fx], corr_v.loc[f, fy]
        q = (x ** 2 + y ** 2)
        fig.add_annotation(x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=2, arrowcolor=SEG_COLORS[0] if q >= 0.5 else "#a3a29c")
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="text", text=[SHORT_LABELS[f]], textposition="top center",
                                 showlegend=False, hovertemplate=f"{f}<br>{fx}: {x:.2f}<br>{fy}: {y:.2f}<br>cos² plan: {q:.2f}<extra></extra>"))
    lam = pca.eigen_table().set_index("Axe")["% inertie"]
    fig.update_layout(template=TEMPLATE, width=520, height=520, xaxis=dict(range=[-1.15, 1.15], zeroline=True,
                      title=f"{fx} ({lam[fx]:.1f} %)"), yaxis=dict(range=[-1.15, 1.15], zeroline=True, scaleanchor="x",
                      title=f"{fy} ({lam[fy]:.1f} %)"), title="Flèches grises : variables mal représentées (cos² < 0,5)")
    cc1.plotly_chart(fig, width="stretch")
    with cc2:
        names_ax = pca.axis_names()
        st.dataframe(pd.DataFrame({"Axe": [f"F{i}" for i in names_ax], "Nom retenu": list(names_ax.values()),
                                   "Variables liées (auto)": [pca.auto_axis_name(i) for i in names_ax]}),
                     hide_index=True, width="stretch")
        st.markdown("**cos² des variables** (qualité de représentation sur chaque axe retenu)")
        cos2 = pca.cos2().iloc[:, :pca.n_axes]
        cos2["Σ axes retenus"] = cos2.sum(axis=1)
        st.dataframe(cos2.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0, vmax=1), width="stretch")
        with st.expander("Corrélations variables / axes et contributions"):
            st.dataframe(corr_v.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")
            st.dataframe(pca.contributions().style.format("{:.1f}"), width="stretch")

# =========================================================================== #
# 3. Modélisation
# =========================================================================== #
with tabs[2]:
    st.header("Module 3 – Modélisation")
    Lz, Lf, coph_z, coph_f = get_linkages(prof, prep_cfg)
    m1, m2, m3 = st.columns(3)
    m1.metric("Corrélation cophénétique – CAH sur Z", fmt(coph_z, 3))
    m2.metric("Corrélation cophénétique – CAH sur axes ACP", fmt(coph_f, 3))
    mixte_info = get_model(prof, (k, "mixte", outlier_mode, z_thr, skew_thr, n_axes, depth)).fit_info_
    m3.metric("Clients déplacés à la consolidation K-Means", f"{mixte_info['deplaces']} "
              f"({100 * mixte_info['part_deplaces']:.1f} %)")
    st.caption("La corrélation cophénétique mesure la fidélité du dendrogramme aux distances d'origine "
               "(> 0,7 : bonne ; ici modérée, signe d'un nuage continu plutôt que de groupes très séparés). "
               "La CAH de Ward ne revient jamais sur une fusion : la consolidation par K-Means corrige ces "
               "affectations sous-optimales.")

    d1, d2 = st.columns([1.4, 1])
    which = d1.radio("Dendrogramme", ["CAH sur les axes ACP (classification mixte)", "CAH sur les variables standardisées"],
                     horizontal=True)
    Lsel = Lf if which.startswith("CAH sur les axes") else Lz
    dd = dendrogram_coords(Lsel, p=30)
    fig = go.Figure()
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#52514e", width=1.5),
                                 hoverinfo="skip", showlegend=False))
    cut = (Lsel[-k, 2] + Lsel[-k + 1, 2]) / 2
    fig.add_hline(y=cut, line_dash="dot", line_color="#e34948", annotation_text=f"coupure k = {k}")
    fig.update_layout(template=TEMPLATE, height=420, title="Dendrogramme (30 dernières fusions, Ward)",
                      xaxis=dict(showticklabels=False, title="groupes (effectifs entre parenthèses au survol)"),
                      yaxis_title="Hauteur de fusion (saut de Ward)")
    d1.plotly_chart(fig, width="stretch")
    d2.markdown("**Hauteurs des dernières fusions**")
    d2.dataframe(cut_heights(Lsel, 9).style.format({"hauteur": "{:.1f}", "saut": "{:.1f}"}), hide_index=True,
                 width="stretch")
    d2.caption("Un grand saut avant une fusion indique qu'on réunit deux groupes hétérogènes : couper juste en dessous.")

    st.subheader("Algorithme supplémentaire : DBSCAN / HDBSCAN")
    Z_in = Z_all[inliers]
    ms_default = 2 * len(FEATURES)
    o1, o2 = st.columns([1, 1.3])
    with o1:
        min_samples = st.slider("minPts (DBSCAN)", 5, 40, ms_default)
        eps_sugg = suggest_eps(Z_in, min_samples)
        eps = st.slider("eps (DBSCAN)", 0.3, 3.0, float(round(eps_sugg, 2)), 0.05,
                        help="Valeur proposée : 95e centile de la distance au minPts-ième voisin")
        min_cluster = st.slider("Taille minimale d'un groupe (HDBSCAN)", 20, 300, 60, 10)
    kd = k_distance_curve(Z_in, min_samples)
    fig = go.Figure(go.Scatter(y=kd, mode="lines", line=dict(color=SEG_COLORS[0], width=2)))
    fig.add_hline(y=eps, line_dash="dot", line_color="#e34948", annotation_text=f"eps = {eps:.2f}")
    fig.update_layout(template=TEMPLATE, height=300, title=f"Graphe des distances au {min_samples}e voisin (triées)",
                      xaxis_title="clients triés", yaxis_title="distance")
    o2.plotly_chart(fig, width="stretch")

    comp, parts = get_comparison(prof, prep_cfg, k, eps, min_samples, min_cluster)
    st.subheader(f"Tableau comparatif des partitions (k = {k} pour les méthodes à k fixé)")
    st.dataframe(comp.style.format({c: "{:.3f}" for c in ["Silhouette", "Davies-Bouldin", "R²", "ARI vs mixte"]}
                                   | {"Calinski-Harabasz": "{:.0f}", "% bruit": "{:.1f}",
                                      "Plus petit segment (%)": "{:.1f}", "Plus grand segment (%)": "{:.1f}"},
                                   na_rep="—"), width="stretch")
    st.caption("Indices calculés dans le même espace (variables standardisées) pour être comparables ; "
               "pour DBSCAN/HDBSCAN, hors points de bruit. Silhouette et R² : plus grand = mieux ; "
               "Davies-Bouldin : plus petit = mieux.")
    st.markdown("**Lecture.** Les méthodes à densité trouvent surtout un grand groupe central et beaucoup de "
                "bruit : la clientèle forme un continuum, sans zones denses séparées. Elles ne fournissent pas "
                "de segments actionnables, mais confirment le repérage des clients atypiques. K-Means et "
                "classification mixte donnent des partitions proches (ARI élevé) ; la mixte est retenue car la "
                "CAH fournit une initialisation déterministe et le dendrogramme justifie k.")
    with st.expander("Tableau croisé : classification mixte × K-Means"):
        ct = pd.crosstab(pd.Series(parts[METHODS["mixte"]], name="mixte"),
                         pd.Series(parts[METHODS["kmeans"]], name="K-Means"))
        st.dataframe(ct, width="stretch")

# =========================================================================== #
# 4. Choix de k et validation
# =========================================================================== #
with tabs[3]:
    st.header("Module 4 – Choix du nombre de segments et validation")
    ind = get_indices(prof, prep_cfg, method, with_options)
    best = best_k_by_index(ind)

    st.subheader(f"1. Indices internes ({METHODS[method]})")
    cols = ["Inertie intra", "R²", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"] + \
           (["Dunn", "Gap", "BIC (GMM)"] if with_options else [])
    nrow = int(np.ceil(len(cols) / 4))
    fig = make_subplots(rows=nrow, cols=4, subplot_titles=cols, vertical_spacing=0.18)
    for i, cname in enumerate(cols):
        rr, cc = i // 4 + 1, i % 4 + 1
        fig.add_trace(go.Scatter(x=ind.index, y=ind[cname], mode="lines+markers", line=dict(color=SEG_COLORS[0], width=2),
                                 marker=dict(size=8), name=cname, showlegend=False,
                                 hovertemplate="k=%{x}<br>%{y:.3f}<extra>" + cname + "</extra>"), row=rr, col=cc)
        fig.add_trace(go.Scatter(x=[k], y=[ind.loc[k, cname]], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=13, color="#eb6834", line=dict(color="white", width=2))), row=rr, col=cc)
    fig.update_layout(template=TEMPLATE, height=280 * nrow, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    i1, i2 = st.columns([1.6, 1])
    i1.dataframe(ind.style.format("{:.3f}").highlight_max(subset=["R²", "Silhouette", "Calinski-Harabasz"], props=HL)
                 .highlight_min(subset=["Davies-Bouldin"], props=HL), width="stretch")
    i2.dataframe(pd.Series(best, name="k recommandé").to_frame(), width="stretch")

    st.subheader("2. Silhouette détaillée de la segmentation retenue")
    s_vals, s_tab = silhouette_detail(Z_all, labels)
    s_tab.index = [f"{s} · {names[s]}" for s in s_tab.index]
    sc1, sc2 = st.columns([1.3, 1])
    fig = go.Figure()
    y0 = 0
    for s in range(k):
        v = np.sort(s_vals[labels == s])
        fig.add_trace(go.Bar(x=v, y=np.arange(y0, y0 + len(v)), orientation="h", marker_color=seg_color(s),
                             marker_line_width=0, name=f"{s} · {names[s]}", width=1.0,
                             hovertemplate="silhouette %{x:.2f}<extra>" + names[s] + "</extra>"))
        y0 += len(v) + 30
    fig.add_vline(x=s_vals.mean(), line_dash="dot", annotation_text=f"moyenne {s_vals.mean():.3f}")
    fig.update_layout(template=TEMPLATE, height=430, bargap=0, yaxis=dict(showticklabels=False),
                      xaxis_title="coefficient de silhouette", legend=dict(orientation="h", y=-0.3))
    sc1.plotly_chart(fig, width="stretch")
    sc2.dataframe(s_tab.style.format({"Silhouette moyenne": "{:.3f}", "% silhouette négative": "{:.1f}"}),
                  width="stretch")
    sc2.caption("Un client à silhouette négative est plus proche, en moyenne, d'un autre segment que du sien : "
                "il est à la frontière. Une part élevée signale un segment aux contours flous.")

    st.subheader("3. Stabilité par bootstrap")
    stab, jac, refs = get_stability(prof, prep_cfg, method, B)
    b1, b2 = st.columns([1.2, 1])
    fig = go.Figure(go.Scatter(x=stab.index, y=stab["ARI moyen"], mode="lines+markers",
                               error_y=dict(type="data", array=stab["ARI écart-type"], color="#a3a29c"),
                               line=dict(color=SEG_COLORS[0], width=2), marker=dict(size=9),
                               hovertemplate="k=%{x}<br>ARI moyen %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[k], y=[stab.loc[k, "ARI moyen"]], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=14, color="#eb6834", line=dict(color="white", width=2))))
    fig.add_hline(y=0.75, line_dash="dot", annotation_text="seuil de stabilité 0,75")
    fig.update_layout(template=TEMPLATE, height=340, showlegend=False, xaxis_title="k",
                      yaxis_title="ARI moyen (± écart-type)", title=f"{B} échantillons bootstrap, réaffectation de tous les clients")
    b1.plotly_chart(fig, width="stretch")
    j = jac[k]
    jt = pd.DataFrame({"Segment": [f"{s} · {names[s]}" for s in range(k)], "Jaccard moyen": j,
                       "Lecture": [jaccard_reading(v) for v in j]})
    b2.dataframe(jt.style.format({"Jaccard moyen": "{:.2f}"}), hide_index=True, width="stretch")
    b2.dataframe(stab.style.format("{:.3f}"), width="stretch")

    st.subheader("4. Indices externes")
    if has_label:
        y = prof["Reachat"].to_numpy()
        ext_rows = []
        for kk in K_RANGE:
            e = external_indices(refs[kk], y[inliers])
            ext_rows.append({"k": kk, **e})
        ext_tab = pd.DataFrame(ext_rows).set_index("k")
        x1, x2 = st.columns([1, 1.2])
        x1.markdown("Étiquette **Réachat dans les "
                    f"{horizon} jours suivants** (jamais utilisée par le clustering).")
        x1.dataframe(ext_tab.style.format("{:.3f}").highlight_max(props=HL), width="stretch")
        rate = label_rate_by_segment(labels, prof["Reachat"])
        rate.index = [f"{s} · {names[s]}" for s in rate.index]
        fig = go.Figure(go.Bar(x=rate["Taux"], y=rate.index, orientation="h",
                               marker_color=[seg_color(s) for s in range(k)],
                               text=rate["Taux"].map(lambda v: f"{v:.0f} %"), textposition="outside",
                               hovertemplate="%{y}<br>taux de réachat %{x:.1f} %<extra></extra>"))
        fig.add_vline(x=100 * y.mean(), line_dash="dot", annotation_text=f"global {100 * y.mean():.0f} %",
                      annotation_position="bottom right")
        fig.update_layout(template=TEMPLATE, height=330, xaxis=dict(title="Taux de réachat (%)", range=[0, 105]),
                          yaxis=dict(autorange="reversed"), title=f"Taux de réachat par segment (k = {k})")
        x2.plotly_chart(fig, width="stretch")
        x2.dataframe(rate.style.format({"Taux": "{:.1f}", "Écart au global (pts)": "{:+.1f}", "Lift": "{:.2f}"}),
                     width="stretch")
        uk = label_rate_by_segment(labels, prof["RoyaumeUni"])["Taux"]
        st.caption("Part de clients britanniques par segment : " +
                   " · ".join(f"{names[s]} {uk.loc[s]:.0f} %" for s in range(k)))
    else:
        st.info("Pas d'étiquette externe (option décochée) : seuls les indices internes, la stabilité et les "
                "critères métier sont utilisés.")

    st.subheader("5. Choix final et justification")
    ka = int(stab["ARI moyen"].idxmax())
    txt = f"""
- **Indices internes** : la silhouette, Calinski-Harabasz et Dunn sont maximaux pour **k = {best['Silhouette (max)']}**,
  le coude de l'inertie est à **k = {best["Coude de l'inertie"]}**{f", le gap statistic (règle 1-s.e.) indique k = {best['Gap statistic (règle 1-s.e.)']} et le BIC du mélange gaussien k = {best['BIC GMM (min)']}" if with_options else ""}.
- **Pourquoi ces désaccords ?** Les clients forment un nuage continu (silhouettes faibles, Dunn proche de 0,
  DBSCAN ne trouve qu'un groupe dense). Silhouette et Calinski-Harabasz récompensent alors la coupure la plus
  grossière (actifs / inactifs), tandis que le R² augmente mécaniquement avec k. Le BIC, qui modélise des
  ellipses gaussiennes, peut préférer plus de groupes que les indices fondés sur des distances.
- **Stabilité** : ARI bootstrap = {stab.loc[k, 'ARI moyen']:.2f} pour k = {k} (le plus stable : k = {ka},
  {stab.loc[ka, 'ARI moyen']:.2f}) ; Jaccard minimal par segment = {j.min():.2f}.
- **Silhouette retenue** : {s_vals.mean():.3f}, {100 * (s_vals < 0).mean():.1f} % de clients à silhouette négative.
"""
    if has_label:
        txt += (f"- **Utilité métier (externe)** : le taux de réachat va de {rate['Taux'].min():.0f} % à "
                f"{rate['Taux'].max():.0f} % selon le segment (global {100 * y.mean():.0f} %). "
                f"Homogénéité = {ext_tab.loc[k, 'Homogénéité']:.3f}.\n")
    txt += f"""- **Critère métier** : k = 2 ou 3 sépare seulement actifs / inactifs / nouveaux et mélange les très gros
  clients avec les réguliers ; au-delà de 6, des segments de moins de 10 % des clients deviennent instables
  et difficiles à animer par des actions distinctes.

➡️ **Choix proposé : k = 5 par classification mixte** — compromis entre stabilité (ARI ≈ 0,8, tous les Jaccard
≥ 0,65), lisibilité (5 actions commerciales distinctes) et pouvoir prédictif du réachat (homogénéité maximale
parmi k = 2..8 sur ces données). Le curseur permet de vérifier en direct l'effet de k.
"""
    st.markdown(txt)

# =========================================================================== #
# 5. Segments expliqués
# =========================================================================== #
with tabs[4]:
    st.header("Module 5 – Explicabilité des segments")

    st.subheader("Nom et action par segment (modifiables)")
    edit_cols = st.columns(min(k, 4))
    new_store = {}
    for s in range(k):
        with edit_cols[s % len(edit_cols)]:
            p = model.profiles_.loc[s]
            st.markdown(f"<div style='border-left:6px solid {seg_color(s)};padding-left:10px'>"
                        f"<b>Segment {s}</b> · {int(p['Effectif'])} clients ({p['% clients']:.0f} %) · "
                        f"CA {p['Part du CA (%)']:.0f} %</div>", unsafe_allow_html=True)
            nm = st.text_input("Nom", value=names[s], key=f"nm_{sig}_{s}")
            ac = st.text_area("Action", value=model.actions[s], key=f"ac_{sig}_{s}", height=120)
            new_store[s] = {"nom": nm, "action": ac}
    if any(new_store[s]["nom"] != names[s] or new_store[s]["action"] != model.actions[s] for s in range(k)):
        st.session_state["names"][sig] = new_store
        st.rerun()

    st.subheader("Profils")
    prof_tab = model.profiles_.copy()
    st.dataframe(prof_tab.style.format({c: "{:,.1f}" for c in prof_tab.columns if c not in ("Nom", "Effectif")}),
                 width="stretch")

    p1, p2 = st.columns([1.3, 1])
    idx = model.index_.copy()
    idx.index = [f"{s} · {names[s]}" for s in idx.index]
    idx.columns = [SHORT_LABELS[c] for c in idx.columns]
    zc = np.log2(idx / 100)  # échelle symétrique autour de 100
    fig = go.Figure(go.Heatmap(z=zc.values, x=idx.columns, y=idx.index, text=idx.round(0).values,
                               texttemplate="%{text}", colorscale=DIVERGING, zmid=0, zmin=-2, zmax=2,
                               colorbar=dict(title="indice", tickvals=[-2, -1, 0, 1, 2],
                                             ticktext=["25", "50", "100", "200", "400"]),
                               hovertemplate="%{y}<br>%{x} : indice %{text}<extra></extra>"))
    fig.update_layout(template=TEMPLATE, height=360, yaxis=dict(autorange="reversed"),
                      title="Indices de profil (moyenne du segment / moyenne globale × 100)")
    p1.plotly_chart(fig, width="stretch")
    share = model.profiles_[["% clients", "Part du CA (%)"]].copy()
    share.index = [f"{s} · {names[s]}" for s in share.index]
    fig = go.Figure()
    for col, colr in [("% clients", "#c3c2b7"), ("Part du CA (%)", SEG_COLORS[0])]:
        fig.add_trace(go.Bar(y=share.index, x=share[col], name=col, orientation="h", marker_color=colr,
                             text=share[col].map(lambda v: f"{v:.0f} %"), textposition="outside"))
    fig.update_layout(template=TEMPLATE, barmode="group", height=360, yaxis=dict(autorange="reversed"),
                      title="Poids en clients vs poids dans le chiffre d'affaires", legend=dict(orientation="h", y=-0.15))
    p2.plotly_chart(fig, width="stretch")

    st.subheader("Carte ACP des segments")
    axn = model.pca.axis_names()
    q1, q2 = st.columns([1, 3])
    na = model.pca.n_axes
    ax_x = q1.selectbox("Axe horizontal", list(range(1, na + 1)), index=0, format_func=lambda i: f"F{i}")
    ax_y = q1.selectbox("Axe vertical", list(range(1, na + 1)), index=1, format_func=lambda i: f"F{i}")
    q1.caption("\n\n".join(f"**F{i}** : {n}" for i, n in axn.items()))
    mapdf = pd.DataFrame({"x": F_all[:, ax_x - 1], "y": F_all[:, ax_y - 1],
                          "Segment": [f"{s} · {names[s]}" for s in labels], "Client": prof.index,
                          "Montant": prof["Montant"].round(0), "Fréquence": prof["Frequence"],
                          "Récence": prof["Recence"]})
    fig = px.scatter(mapdf, x="x", y="y", color="Segment", template=TEMPLATE, opacity=0.65, height=560,
                     color_discrete_sequence=SEG_COLORS, hover_data=["Client", "Montant", "Fréquence", "Récence"],
                     category_orders={"Segment": [f"{s} · {names[s]}" for s in range(k)]},
                     labels={"x": f"F{ax_x} – {axn[ax_x]}", "y": f"F{ax_y} – {axn[ax_y]}"})
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
    cen = np.vstack([F_all[labels == s].mean(axis=0) for s in range(k)])
    fig.add_trace(go.Scatter(x=cen[:, ax_x - 1], y=cen[:, ax_y - 1], mode="markers+text", name="Centres",
                             text=[names[s] for s in range(k)], textposition="top center",
                             marker=dict(symbol="x", size=16, color="#0b0b0b", line=dict(width=2, color="white"))))
    q2.plotly_chart(fig, width="stretch")

    st.subheader(f"Arbre de décision de substitution (profondeur {depth})")
    ti = model.tree_info_
    t1, t2, t3 = st.columns(3)
    t1.metric("Fidélité (accord avec les segments)", f"{100 * ti['fidelite']:.1f} %")
    t2.metric("Fidélité en validation croisée", f"{100 * ti['fidelite_cv']:.1f} %")
    t3.metric("Nombre de règles (feuilles)", ti["feuilles"])
    st.caption("L'arbre est appris sur les variables brutes pour prédire le segment : ses seuils se lisent "
               "directement en jours, commandes et livres sterling. Sa fidélité indique quelle part de la "
               "segmentation ces règles simples reproduisent.")
    for s in range(k):
        rs = model.rules_[model.rules_["Segment"] == s]
        with st.container(border=True):
            st.markdown(f"<span style='color:{seg_color(s)};font-size:1.3em'>●</span> **{names[s]}** — "
                        f"{model.actions[s]}", unsafe_allow_html=True)
            if rs.empty:
                st.markdown("_Aucune feuille de l'arbre ne prédit ce segment à cette profondeur : il est décrit "
                            "par ses médianes et indices de profil._")
            for rule in rs["Règle"]:
                st.markdown(f"- {rule}")

# =========================================================================== #
# 6. Nouveau client et API
# =========================================================================== #
with tabs[5]:
    st.header("Explication d'une affectation et affectation d'un nouveau client")
    mode = st.radio("Client", ["Client existant", "Nouveau client (saisie)"], horizontal=True)
    if mode == "Client existant":
        default_id = int(prof.index[0])
        cid = st.selectbox("Identifiant client", prof.index.tolist(), index=0)
        x = prof.loc[cid, FEATURES]
    else:
        gm = model.global_median_
        cols_in = st.columns(4)
        vals = {}
        for i, f in enumerate(FEATURES):
            step = 1.0 if f in ("Recence", "Frequence", "NbProduits", "Anciennete") else 10.0
            vals[f] = cols_in[i % 4].number_input(FEATURE_LABELS[f], min_value=0.0, value=float(round(gm[f], 1)),
                                                 step=step if f != "TauxRetour" else 0.5)
        x = pd.Series(vals)
    e = model.explain(x)

    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(f"<div style='border-left:8px solid {seg_color(e['segment'])};padding:6px 12px'>"
                f"<div style='font-size:0.85em;color:#52514e'>Segment affecté</div>"
                f"<div style='font-size:1.4em;font-weight:600'>{e['segment']} · {e['nom']}</div></div>",
                unsafe_allow_html=True)
    r2.metric("Distance au centre", fmt(e["distance_centre"]), help=f"Seuil atypique : {e['seuil_atypique']:.2f}")
    r3.metric("Segment suivant", e["nom_segment_suivant"], f"marge {e['marge']:.2f}", delta_color="off")
    r4.metric("Client atypique", "⚠️ OUI" if e["atypique"] else "Non")
    if e["atypique"]:
        st.warning("Client atypique : " + " ; ".join(e["raisons_atypique"]) +
                   ". Son affectation est à confirmer par un commercial.")
    st.success(f"**Action recommandée** : {e['action']}")
    st.markdown(f"**Règle de l'arbre** : {e['regle_arbre']} → « {names.get(e['segment_arbre'], '')} »"
                + ("" if e["segment_arbre"] == e["segment"] else "  _(l'arbre simplifié ne donne pas le même segment)_"))

    v1, v2 = st.columns([1.1, 1])
    with v1:
        st.markdown("**Variables qui l'éloignent le plus du profil moyen**")
        for vinfo in e["variables_principales"]:
            st.markdown(f"- {vinfo['phrase']}")
        vd = pd.DataFrame(e["variables"])
        vd = vd.set_index("libelle").loc[[SHORT_LABELS[f] for f in FEATURES]]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=vd.index, x=vd["z_score"], orientation="h", name="Client (z-score)",
                             marker_color=[SEG_COLORS[7] if v > 0 else SEG_COLORS[0] for v in vd["z_score"]],
                             hovertemplate="%{y} : z = %{x:.2f}<extra></extra>"))
        cz = model.centers_z_[e["segment"]]
        fig.add_trace(go.Scatter(y=vd.index, x=cz, mode="markers", name=f"Centre du segment « {e['nom']} »",
                                 marker=dict(symbol="diamond", size=12, color="#0b0b0b", line=dict(color="white", width=2))))
        fig.add_vline(x=0, line_color="#52514e")
        fig.update_layout(template=TEMPLATE, height=380, xaxis_title="écart au client moyen (z-score après log)",
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with v2:
        st.markdown("**Détail par variable**")
        show = vd[["valeur", "mediane_segment", "mediane_globale", "z_score", "part_distance_segment"]].rename(columns={
            "valeur": "Client", "mediane_segment": "Médiane segment", "mediane_globale": "Médiane globale",
            "z_score": "z", "part_distance_segment": "Part de la distance au centre"})
        st.dataframe(show.style.format({"Client": "{:,.1f}", "Médiane segment": "{:,.1f}", "Médiane globale": "{:,.1f}",
                                        "z": "{:+.2f}", "Part de la distance au centre": "{:.0%}"}), width="stretch")
        dist_df = pd.DataFrame({"Segment": [f"{s} · {names[s]}" for s in e["distances"]],
                                "Distance": list(e["distances"].values())})
        fig = go.Figure(go.Bar(x=dist_df["Distance"], y=dist_df["Segment"], orientation="h",
                               marker_color=[seg_color(s) for s in e["distances"]]))
        fig.update_layout(template=TEMPLATE, height=260, yaxis=dict(autorange="reversed"),
                          title="Distance aux centres des segments", xaxis_title="distance (espace du modèle)")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.header("Module 6 – Exposer le modèle par une API (FastAPI)")
    s1, s2 = st.columns([1, 1.4])
    with s1:
        st.markdown(f"Le modèle courant (k = {k}, {METHODS[method]}, noms et actions saisis) peut être enregistré "
                    "puis servi par l'API.")
        if st.button("💾 Enregistrer ce modèle pour l'API", type="primary"):
            meta = {kk: str(vv) for kk, vv in prof_info.items()}
            model.meta.update(meta)
            path = model.save(MODEL_PATH)
            st.success(f"Modèle enregistré : `{path.name}`. Redémarrez l'API ou appelez `POST /model/reload`.")
        if MODEL_PATH.exists():
            st.caption(f"Fichier : models/{MODEL_PATH.name}")
    with s2:
        st.markdown("Lancer l'API puis ouvrir la documentation interactive sur http://127.0.0.1:8000/docs :")
        st.code("uvicorn api:app --reload --port 8000", language="bash")
        example = {f: float(round(x[f], 2)) for f in FEATURES}
        st.code("curl -X POST http://127.0.0.1:8000/predict -H \"Content-Type: application/json\" \\\n  -d '"
                + json.dumps(example) + "'", language="bash")
