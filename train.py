"""Entraîne le modèle de segmentation retenu et le sauvegarde pour l'API.

Usage :
    python train.py                 # k=5, classification mixte, extrêmes écrêtés
    python train.py --k 4 --method kmeans --outliers a_part
    python train.py --reports       # exporte aussi les tableaux d'analyse dans reports/
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from seg.data import FEATURES, ROOT, load_profile
from seg.model import MODEL_PATH, SegmentationConfig, SegmentationModel
from seg.validation import (bootstrap_stability, external_indices, internal_indices,
                            label_rate_by_segment, silhouette_detail)
from seg.clustering import fit_partition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", default="mixte", choices=["mixte", "kmeans", "cah"])
    ap.add_argument("--outliers", default="ecreter", choices=["ecreter", "conserver", "a_part", "dbscan"])
    ap.add_argument("--axes", type=int, default=None)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--reports", action="store_true", help="exporte les tableaux dans reports/")
    ap.add_argument("--bootstrap", type=int, default=20)
    args = ap.parse_args()

    print("Chargement des données…")
    tx, report, prof, info = load_profile()
    print(f"  {info['n_clients']} clients, fenêtre d'observation jusqu'au {info['fin_observation']}")

    cfg = SegmentationConfig(k=args.k, method=args.method, outlier_mode=args.outliers,
                             n_axes=args.axes, tree_depth=args.depth)
    model = SegmentationModel(cfg).fit(prof, meta={k: str(v) for k, v in info.items()})
    path = model.save(MODEL_PATH)
    print(f"Modèle sauvegardé : {path}")
    print(f"  axes ACP retenus : {model.pca.n_axes}, métriques : "
          + ", ".join(f"{k}={v:.3f}" for k, v in model.metrics_.items()))
    print(f"  arbre de substitution : fidélité {model.tree_info_['fidelite']:.1%} "
          f"(validation croisée {model.tree_info_['fidelite_cv']:.1%})")
    for s in model.summary():
        print(f"  [{s['segment']}] {s['nom']:<38} {s['effectif']:>5} clients  "
              f"{s['part_clients_pct']:5.1f} %  CA {s['part_ca_pct']:5.1f} %")

    if args.reports:
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        pp = model.preprocessor
        Zdf = pp.transform(prof[FEATURES])
        inl = model.inliers_
        Z, F = Zdf.to_numpy()[inl], model.pca.transform(Zdf)[inl]
        pp.skewness_.to_csv(out / "transformations.csv", encoding="utf-8-sig")
        model.pca.eigen_table().to_csv(out / "acp_valeurs_propres.csv", encoding="utf-8-sig", index=False)
        model.pca.correlations().to_csv(out / "acp_correlations.csv", encoding="utf-8-sig")
        tab = internal_indices(cfg.method, Z, F, seed=cfg.seed)
        tab.to_csv(out / "indices_internes.csv", encoding="utf-8-sig")
        refs = {k: fit_partition(cfg.method, Z, F, k, cfg.seed)[0] for k in tab.index}
        st, jac = bootstrap_stability(cfg.method, Z, F, refs, B=args.bootstrap, seed=cfg.seed)
        st.to_csv(out / "stabilite_ari.csv", encoding="utf-8-sig")
        lab = model.labels_
        _, sil = silhouette_detail(Zdf.to_numpy(), lab)
        sil.to_csv(out / "silhouette_par_segment.csv", encoding="utf-8-sig")
        model.profiles_.to_csv(out / "profils_segments.csv", encoding="utf-8-sig")
        model.index_.to_csv(out / "indices_profil.csv", encoding="utf-8-sig")
        model.rules_.to_csv(out / "regles_arbre.csv", encoding="utf-8-sig", index=False)
        ext = {}
        if "Reachat" in prof:
            ext = external_indices(lab, prof["Reachat"].to_numpy())
            label_rate_by_segment(lab, prof["Reachat"]).to_csv(out / "reachat_par_segment.csv", encoding="utf-8-sig")
        json.dump({"externes_reachat": ext, "jaccard_k_retenu": jac.get(cfg.k, []).tolist()
                   if hasattr(jac.get(cfg.k, []), "tolist") else [],
                   "metriques": model.metrics_, "arbre": model.tree_info_},
                  open(out / "synthese.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"Rapports exportés dans {out}")
        print(tab.round(3).to_string())
        print(st.round(3).to_string())


if __name__ == "__main__":
    main()
