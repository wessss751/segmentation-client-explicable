"""Tests de fumée : l'application tourne sans erreur pour plusieurs configurations.

    python -m pytest tests -q        (ou : python tests/test_app.py)
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

CONFIGS = [
    {},  # défaut : mixte, k = 5, écrêtage
    {"k": 3, "method": "kmeans", "outliers": "a_part"},
    {"k": 6, "method": "cah", "outliers": "dbscan"},
    {"k": 4, "method": "mixte", "outliers": "conserver", "label": False},
]


def run(cfg: dict):
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    if "label" in cfg:
        at.checkbox[0].set_value(cfg["label"])
    if "outliers" in cfg:
        at.selectbox[0].select_index(["ecreter", "conserver", "a_part", "dbscan"].index(cfg["outliers"]))
    if "method" in cfg:
        at.selectbox[1].select_index(["mixte", "kmeans", "cah"].index(cfg["method"]))
    if "k" in cfg:
        [s for s in at.slider if s.label == "Nombre de segments k"][0].set_value(cfg["k"])
    at.checkbox[2].set_value(False)  # indices optionnels désactivés pour aller vite
    at.run()
    assert not at.exception, f"{cfg}: {at.exception}"
    return at


def test_configs():
    for cfg in CONFIGS:
        run(cfg)


if __name__ == "__main__":
    for cfg in CONFIGS:
        at = run(cfg)
        print("OK", cfg, "-", len(at.markdown), "blocs markdown")
