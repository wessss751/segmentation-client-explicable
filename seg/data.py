"""Module 1 - Données et profil client.

Jeu de données : Online Retail (UCI, id 352) - factures d'un distributeur
britannique en ligne entre le 01/12/2010 et le 09/12/2011.

Profil client construit à partir des transactions brutes :
    RFM classique  : Recence, Frequence, Montant
    + indicateurs  : PanierMoyen, NbProduits, Anciennete, TauxRetour

Étiquette externe (non utilisée par le clustering) : le jeu n'en contient pas.
On en construit une indépendante par découpage temporel : le profil est
calculé sur une fenêtre d'observation, et l'étiquette « Reachat » vaut 1 si le
client achète à nouveau dans les `horizon_days` jours qui suivent. Elle sert
uniquement à évaluer l'utilité des segments (indices externes).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_XLSX = RAW_DIR / "Online Retail.xlsx"
CACHE_PARQUET = DATA_DIR / "online_retail.parquet"
UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

# Variables du profil client (ordre utilisé partout dans l'application)
FEATURES = [
    "Recence",
    "Frequence",
    "Montant",
    "PanierMoyen",
    "NbProduits",
    "Anciennete",
    "TauxRetour",
]

FEATURE_LABELS = {
    "Recence": "Récence (jours depuis le dernier achat)",
    "Frequence": "Fréquence (nombre de commandes)",
    "Montant": "Montant net dépensé (£)",
    "PanierMoyen": "Panier moyen (£ par commande)",
    "NbProduits": "Nombre de produits distincts achetés",
    "Anciennete": "Ancienneté (jours depuis le 1er achat)",
    "TauxRetour": "Taux de retour (% du montant acheté annulé)",
}

FEATURE_UNITS = {
    "Recence": "jours",
    "Frequence": "commandes",
    "Montant": "£",
    "PanierMoyen": "£",
    "NbProduits": "produits",
    "Anciennete": "jours",
    "TauxRetour": "%",
}

# Colonnes exclues du clustering (servent à l'évaluation / description)
EXTERNAL_COLUMNS = ["Reachat", "Pays", "RoyaumeUni"]

# Codes article qui ne sont pas des produits (frais, port, ajustements…)
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "C2", "D", "CRUK", "BANK CHARGES", "AMAZONFEE",
    "PADS", "B", "S", "m",
}

# Renommage pour accepter aussi Online Retail II ou un export proche
COLUMN_ALIASES = {
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID",
    "Customer_ID": "CustomerID",
}
REQUIRED_COLUMNS = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


# --------------------------------------------------------------------------- #
# Téléchargement et chargement
# --------------------------------------------------------------------------- #
def download_dataset(force: bool = False, timeout: int = 120) -> Path:
    """Télécharge le fichier Excel depuis l'UCI s'il n'est pas déjà présent."""
    if RAW_XLSX.exists() and not force:
        return RAW_XLSX
    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(UCI_URL, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        RAW_XLSX.write_bytes(zf.read(member))
    return RAW_XLSX


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le fichier : " + ", ".join(missing)
            + ". Colonnes attendues : " + ", ".join(REQUIRED_COLUMNS)
        )
    df = df[REQUIRED_COLUMNS + [c for c in ["Description"] if c in df.columns]].copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["Country"] = df["Country"].astype(str)
    if "Description" in df.columns:
        df["Description"] = df["Description"].astype(str)
    return df


def load_transactions(source: str | Path | io.BytesIO | None = None, filename: str | None = None) -> pd.DataFrame:
    """Charge les transactions.

    source=None  -> fichier UCI (téléchargé si besoin, mis en cache en parquet)
    source=chemin ou buffer -> import d'un fichier .xlsx / .csv
    """
    if source is None:
        if CACHE_PARQUET.exists():
            return pd.read_parquet(CACHE_PARQUET)
        path = download_dataset()
        df = _standardize_columns(pd.read_excel(path, engine="openpyxl"))
        try:
            df.to_parquet(CACHE_PARQUET, index=False)
        except Exception:  # pyarrow absent : pas de cache, ce n'est pas bloquant
            pass
        return df

    name = (filename or str(source)).lower()
    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(source, engine="openpyxl")
    else:
        raw = pd.read_csv(source, encoding="latin-1", sep=None, engine="python")
    return _standardize_columns(raw)


# --------------------------------------------------------------------------- #
# Nettoyage
# --------------------------------------------------------------------------- #
def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nettoie les lignes de facture et renvoie (transactions, rapport)."""
    steps = [("Lignes brutes", len(df))]

    out = df.dropna(subset=["CustomerID", "Quantity", "UnitPrice", "InvoiceDate"]).copy()
    steps.append(("Sans identifiant client ou valeur manquante retirées", len(out)))

    out = out.drop_duplicates()
    steps.append(("Doublons exacts retirés", len(out)))

    is_product = out["StockCode"].str.match(r"^\d{5}") & ~out["StockCode"].isin(NON_PRODUCT_CODES)
    out = out[is_product]
    steps.append(("Lignes hors produit retirées (port, frais, ajustements)", len(out)))

    out = out[out["UnitPrice"] > 0]
    steps.append(("Prix unitaire nul ou négatif retirés", len(out)))

    out["CustomerID"] = out["CustomerID"].astype(int)
    out["Annulation"] = out["InvoiceNo"].str.upper().str.startswith("C") | (out["Quantity"] < 0)
    out["Amount"] = out["Quantity"] * out["UnitPrice"]

    report = pd.DataFrame(steps, columns=["Étape", "Lignes restantes"])
    report["Lignes retirées"] = (-report["Lignes restantes"].diff()).fillna(0).astype(int)
    return out.reset_index(drop=True), report


# --------------------------------------------------------------------------- #
# Profil client
# --------------------------------------------------------------------------- #
@dataclass
class ProfileConfig:
    obs_end: pd.Timestamp | None = None  # fin (exclue) de la fenêtre d'observation
    horizon_days: int = 90  # fenêtre de réachat pour l'étiquette externe
    use_label: bool = True  # False : toute la période, pas d'étiquette Reachat
    min_net_amount: float = 0.0  # clients au montant net <= seuil retirés
    info: dict = field(default_factory=dict)


def default_obs_end(tx: pd.DataFrame, horizon_days: int = 90) -> pd.Timestamp:
    last = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    return last - pd.Timedelta(days=horizon_days)


def profile_from_transactions(tx: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """Calcule les 7 variables du profil pour chaque client présent dans tx.

    Utilisée à l'entraînement et par l'API pour un nouveau client.
    """
    if "Annulation" not in tx.columns:
        tx = tx.copy()
        tx["Annulation"] = tx["InvoiceNo"].astype(str).str.upper().str.startswith("C") | (tx["Quantity"] < 0)
    if "Amount" not in tx.columns:
        tx = tx.copy()
        tx["Amount"] = tx["Quantity"] * tx["UnitPrice"]

    buys = tx[~tx["Annulation"]]
    rets = tx[tx["Annulation"]]

    g = buys.groupby("CustomerID")
    prof = pd.DataFrame({
        "DernierAchat": g["InvoiceDate"].max(),
        "PremierAchat": g["InvoiceDate"].min(),
        "Frequence": g["InvoiceNo"].nunique(),
        "MontantBrut": g["Amount"].sum(),
        "NbProduits": g["StockCode"].nunique(),
    })
    ret_amount = rets.groupby("CustomerID")["Amount"].sum().abs()
    prof["MontantRetour"] = ret_amount.reindex(prof.index).fillna(0.0)

    prof["Recence"] = (ref_date - prof["DernierAchat"]).dt.days.clip(lower=0)
    prof["Anciennete"] = (ref_date - prof["PremierAchat"]).dt.days.clip(lower=0)
    prof["Montant"] = prof["MontantBrut"] - prof["MontantRetour"]
    prof["PanierMoyen"] = prof["MontantBrut"] / prof["Frequence"]
    prof["TauxRetour"] = 100 * (prof["MontantRetour"] / prof["MontantBrut"]).clip(0, 1)

    country = tx.groupby("CustomerID")["Country"].agg(lambda s: s.mode().iat[0])
    prof["Pays"] = country.reindex(prof.index)
    return prof


def build_customer_profile(tx: pd.DataFrame, cfg: ProfileConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Construit la table clients (une ligne par client).

    Renvoie (profil, infos). Le profil contient FEATURES + colonnes externes.
    """
    cfg = cfg or ProfileConfig()
    data_start = tx["InvoiceDate"].min()
    data_end = tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    if cfg.use_label:
        obs_end = cfg.obs_end if cfg.obs_end is not None else default_obs_end(tx, cfg.horizon_days)
        obs_end = pd.Timestamp(obs_end)
        horizon_end = min(obs_end + pd.Timedelta(days=cfg.horizon_days), data_end)
    else:
        obs_end, horizon_end = data_end, data_end

    obs = tx[tx["InvoiceDate"] < obs_end]
    prof = profile_from_transactions(obs, ref_date=obs_end)

    n_before = len(prof)
    prof = prof[prof["Montant"] > cfg.min_net_amount]
    n_removed = n_before - len(prof)

    if cfg.use_label:
        fut = tx[(tx["InvoiceDate"] >= obs_end) & (tx["InvoiceDate"] < horizon_end) & (~tx["Annulation"])]
        buyers = set(fut["CustomerID"].unique())
        prof["Reachat"] = prof.index.isin(buyers).astype(int)
    prof["RoyaumeUni"] = (prof["Pays"] == "United Kingdom").astype(int)

    keep = FEATURES + [c for c in EXTERNAL_COLUMNS if c in prof.columns]
    prof = prof[keep].copy()
    prof.index.name = "CustomerID"

    info = {
        "periode_donnees": (data_start.date(), (data_end - pd.Timedelta(days=1)).date()),
        "fin_observation": obs_end.date(),
        "fin_horizon": horizon_end.date() if cfg.use_label else None,
        "horizon_jours": cfg.horizon_days if cfg.use_label else None,
        "n_clients": len(prof),
        "n_clients_montant_negatif_retires": int(n_removed),
        "taux_reachat": float(prof["Reachat"].mean()) if "Reachat" in prof else None,
    }
    return prof, info


def load_profile(use_label: bool = True, horizon_days: int = 90, source=None, filename=None):
    """Raccourci : transactions -> nettoyage -> profil."""
    tx = load_transactions(source, filename)
    tx, report = clean_transactions(tx)
    prof, info = build_customer_profile(tx, ProfileConfig(horizon_days=horizon_days, use_label=use_label))
    return tx, report, prof, info
