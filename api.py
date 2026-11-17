"""Module 6 - API FastAPI du modèle de segmentation.

Lancement :  uvicorn api:app --reload --port 8000
Documentation interactive : http://127.0.0.1:8000/docs

Le modèle est produit par `python train.py` ou par le bouton « Enregistrer ce
modèle pour l'API » de l'application Streamlit (models/segmentation_model.joblib).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from seg.data import FEATURES, profile_from_transactions
from seg.model import MODEL_PATH, SegmentationModel

app = FastAPI(
    title="API de segmentation client explicable",
    description="Affecte un client à son segment (Online Retail, UCI), explique l'affectation "
                "et propose l'action commerciale associée.",
    version="1.0.0",
)

_state: dict = {"model": None}


def get_model() -> SegmentationModel:
    if _state["model"] is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Aucun modèle entraîné : lancez `python train.py`.")
        _state["model"] = SegmentationModel.load(MODEL_PATH)
    return _state["model"]


# --------------------------------------------------------------------------- #
# Schémas
# --------------------------------------------------------------------------- #
class Customer(BaseModel):
    Recence: float = Field(..., ge=0, description="Jours depuis le dernier achat", examples=[30])
    Frequence: float = Field(..., ge=1, description="Nombre de commandes", examples=[4])
    Montant: float = Field(..., gt=0, description="Montant net dépensé (£)", examples=[1200])
    PanierMoyen: float = Field(..., gt=0, description="Montant moyen par commande (£)", examples=[300])
    NbProduits: float = Field(..., ge=1, description="Nombre de produits distincts", examples=[45])
    Anciennete: float = Field(..., ge=0, description="Jours depuis le premier achat", examples=[200])
    TauxRetour: float = Field(0.0, ge=0, le=100, description="% du montant acheté annulé", examples=[0])


class Transaction(BaseModel):
    InvoiceNo: str = Field(..., examples=["536365"])
    StockCode: str = Field(..., examples=["85123A"])
    Quantity: float = Field(..., examples=[6])
    InvoiceDate: dt.datetime = Field(..., examples=["2011-08-01T10:00:00"])
    UnitPrice: float = Field(..., gt=0, examples=[2.55])


class TransactionsIn(BaseModel):
    transactions: list[Transaction]
    date_reference: dt.datetime | None = Field(
        None, description="Date d'analyse pour la récence/ancienneté (défaut : lendemain du dernier achat)")


class VariableExplanation(BaseModel):
    variable: str
    valeur: float
    z_score: float
    mediane_segment: float
    mediane_globale: float
    part_distance_segment: float
    phrase: str


class Prediction(BaseModel):
    segment: int
    nom: str
    action: str
    distance_centre: float
    atypique: bool
    raisons_atypique: list[str]
    segment_suivant: str
    marge: float
    regle_arbre: str
    variables_principales: list[VariableExplanation]
    profil: dict | None = None


def _predict(model: SegmentationModel, x: dict) -> Prediction:
    e = model.explain(x)
    return Prediction(
        segment=e["segment"], nom=e["nom"], action=e["action"], distance_centre=round(e["distance_centre"], 4),
        atypique=e["atypique"], raisons_atypique=e["raisons_atypique"], segment_suivant=e["nom_segment_suivant"],
        marge=round(e["marge"], 4), regle_arbre=e["regle_arbre"],
        variables_principales=[VariableExplanation(**{k: v[k] for k in VariableExplanation.model_fields})
                               for v in e["variables_principales"]],
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", tags=["info"])
def root():
    return {"message": "API de segmentation client explicable", "documentation": "/docs",
            "routes": ["/health", "/model/info", "/segments", "/predict", "/predict/batch",
                       "/predict/transactions", "/model/reload"]}


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "modele_disponible": MODEL_PATH.exists()}


@app.get("/model/info", tags=["info"])
def model_info():
    m = get_model()
    return {
        "configuration": m.config_dict(),
        "variables": FEATURES,
        "variables_log": m.preprocessor.log_vars,
        "axes_acp": m.pca.n_axes,
        "noms_axes": m.pca.axis_names(),
        "metriques": m.metrics_,
        "arbre_substitution": m.tree_info_,
        "meta": m.meta,
    }


@app.post("/model/reload", tags=["info"])
def reload_model():
    _state["model"] = None
    m = get_model()
    return {"status": "rechargé", "k": m.config.k, "entraine_le": m.meta.get("entraine_le")}


@app.get("/segments", tags=["segments"])
def segments():
    return get_model().summary()


@app.get("/segments/{segment_id}", tags=["segments"])
def segment(segment_id: int):
    m = get_model()
    if not 0 <= segment_id < m.config.k:
        raise HTTPException(404, f"Segment inconnu (0 à {m.config.k - 1}).")
    return m.summary()[segment_id]


@app.post("/predict", response_model=Prediction, tags=["affectation"])
def predict(customer: Customer):
    """Affecte un client décrit par son profil (7 variables) et explique l'affectation."""
    return _predict(get_model(), customer.model_dump())


@app.post("/predict/batch", response_model=list[Prediction], tags=["affectation"])
def predict_batch(customers: list[Customer]):
    m = get_model()
    return [_predict(m, c.model_dump()) for c in customers]


@app.post("/predict/transactions", response_model=Prediction, tags=["affectation"])
def predict_from_transactions(payload: TransactionsIn):
    """Construit le profil d'un nouveau client à partir de ses lignes de facture brutes
    (factures d'annulation : InvoiceNo commençant par « C » ou quantité négative)."""
    if not payload.transactions:
        raise HTTPException(422, "Liste de transactions vide.")
    tx = pd.DataFrame([t.model_dump() for t in payload.transactions])
    tx["CustomerID"] = 0
    tx["Country"] = "?"
    tx["InvoiceDate"] = pd.to_datetime(tx["InvoiceDate"]).dt.tz_localize(None)
    ref = (pd.Timestamp(payload.date_reference).tz_localize(None) if payload.date_reference
           else tx["InvoiceDate"].max().normalize() + pd.Timedelta(days=1))
    prof = profile_from_transactions(tx, ref)
    if prof.empty or prof["Montant"].iloc[0] <= 0:
        raise HTTPException(422, "Aucun achat net positif : impossible de profiler ce client.")
    x = prof.iloc[0][FEATURES].astype(float).to_dict()
    pred = _predict(get_model(), x)
    pred.profil = {k: round(v, 2) for k, v in x.items()}
    return pred
