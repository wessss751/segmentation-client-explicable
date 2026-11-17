# Projet Challenge 3 – Application web de segmentation client explicable

Data Mining · ENSI · 2026-2027 — Jeu de données n°1 du catalogue : **Online Retail (UCI)**
(factures d'un distributeur britannique en ligne, 01/12/2010 → 09/12/2011, 541 909 lignes).

L'application construit un profil RFM enrichi par client, le résume par une ACP normée,
compare plusieurs algorithmes de clustering, choisit et justifie le nombre de segments,
les nomme, les explique, propose une action pour chacun et affecte un nouveau client à son
segment (interface Streamlit + API FastAPI).

---

## 1. Installation et lancement

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac : source .venv/bin/activate)
pip install -r requirements.txt
```

| Commande | Rôle |
|---|---|
| `streamlit run app.py` | Application web interactive → http://localhost:8501 |
| `python train.py --reports` | Entraîne le modèle retenu, le sauvegarde dans `models/` et exporte les tableaux dans `reports/` |
| `uvicorn api:app --reload --port 8000` | API → documentation interactive http://127.0.0.1:8000/docs |
| `python -m pytest tests -q` | Test de fumée de l'application sur 4 configurations |

Au premier lancement, le fichier est **téléchargé automatiquement** depuis l'UCI
(`data/raw/Online Retail.xlsx`) puis mis en cache en parquet. Un fichier de factures
(.xlsx / .csv, même format ou Online Retail II) peut aussi être **importé** depuis la barre latérale.

## 2. Organisation du code

```
segmentation_app/
├── app.py              interface Streamlit (6 onglets = 6 modules)
├── api.py              API FastAPI
├── train.py            entraînement en ligne de commande + export des rapports
├── seg/
│   ├── data.py           Module 1 : téléchargement, nettoyage, profil client, étiquette externe
│   ├── preprocessing.py  Module 2 : log, standardisation, valeurs extrêmes
│   ├── pca.py            Module 2 : ACP normée, règles du nombre d'axes, noms des axes
│   ├── clustering.py     Module 3 : K-Means, CAH Ward, classification mixte, DBSCAN, HDBSCAN
│   ├── validation.py     Module 4 : indices internes, bootstrap, indices externes
│   ├── explain.py        Module 5 : profils, noms/actions, arbre de substitution
│   └── model.py          modèle complet (fit / predict / explain / save)
├── models/             modèle sauvegardé (joblib) utilisé par l'API
├── reports/            tableaux exportés par train.py --reports
└── tests/
```

## 3. Démarche et résultats

### Module 1 – Données et profil client
Nettoyage : lignes sans `CustomerID` (135 080), doublons exacts (5 225), codes non-produits
(port `POST`, frais `BANK CHARGES`, `AMAZONFEE`, ajustements `M`, `D`…), prix nuls.
Les annulations (factures `C…`) sont conservées pour calculer le taux de retour.

**Profil client (7 variables)** : RFM classique + 4 indicateurs ajoutés (le sujet en demande au moins deux).

| Variable | Définition |
|---|---|
| Recence | jours entre le dernier achat et la date d'analyse |
| Frequence | nombre de factures distinctes |
| Montant | montant net (achats − retours), £ |
| PanierMoyen | montant brut / nombre de commandes |
| NbProduits | nombre de références distinctes achetées (largeur de gamme) |
| Anciennete | jours depuis le premier achat |
| TauxRetour | % du montant acheté annulé |

**Étiquette externe.** Online Retail n'en a pas. On en crée une *indépendante* par découpage
temporel : le profil est calculé sur les données **avant le 11/09/2011** (3 360 clients) et
`Reachat = 1` si le client achète à nouveau dans les **90 jours suivants** (57 % des clients).
Cette variable, ainsi que le pays, est **exclue du clustering** et sert uniquement à évaluer
l'utilité des segments.

### Module 2 – Préparation et ACP
- `log(1+x)` sur les variables positives d'asymétrie > 1 : Frequence (9,2 → 1,4), Montant
  (17,4 → 0,4), PanierMoyen (13,5 → 0,1), NbProduits (6,2 → −0,2), TauxRetour (7,2 → 2,1).
  Récence (0,8) et Ancienneté (−0,5) restent sur leur échelle.
- Standardisation de toutes les variables.
- **Valeurs extrêmes : écrêtées à ±3 écarts-types** (154 clients, 4,6 %, surtout des
  grossistes et des retours massifs). Justification : tous les clients restent segmentés mais
  ne tirent plus les centres ; ils sont repérés par le drapeau « client atypique ».
  Les 3 autres options (conserver, suivre à part, isoler par DBSCAN) sont disponibles dans l'application.
- **ACP normée** : λ = 3,24 / 1,37 / 0,98 / 0,85… Kaiser strict = 2 axes (65,8 %), mais
  λ₃ = 0,98 ≈ 1 et F3 a un sens clair → **3 axes retenus (79,9 % d'inertie)**.

| Axe | Nom | Variables (corrélation) |
|---|---|---|
| F1 (46 %) | Valeur et intensité d'achat | Montant 0,96 · Fréquence 0,84 · NbProduits 0,84 · Récence −0,56 |
| F2 (20 %) | Ancienneté sans achat récent | Ancienneté 0,88 · Récence 0,72 |
| F3 (14 %) | Gros paniers peu fréquents | PanierMoyen 0,74 · Fréquence −0,41 |

### Module 3 – Modélisation
K-Means (20 initialisations), CAH de Ward (dendrogramme, corrélation cophénétique 0,43 sur Z,
0,54 sur les axes ACP), **classification mixte** (CAH de Ward sur F1-F3 → consolidation K-Means
initialisée aux centres de la CAH), DBSCAN (eps par graphe des k-distances) et HDBSCAN.
DBSCAN/HDBSCAN trouvent un grand groupe dense et du bruit : la clientèle forme un continuum,
les méthodes par densité servent surtout à repérer les atypiques. Le tableau comparatif
(silhouette, CH, DB, R², tailles, ARI entre partitions) est dans l'onglet 3.

### Module 4 – Choix de k (classification mixte)

| k | R² | Silhouette | CH | DB | ARI bootstrap |
|---|---|---|---|---|---|
| 2 | 0,32 | **0,282** | **1541** | **1,36** | **0,98** |
| 3 | 0,44 | 0,257 | 1298 | 1,40 | 0,95 |
| 4 | 0,50 | 0,209 | 1107 | 1,50 | 0,74 |
| **5** | **0,54** | 0,205 | 997 | 1,46 | **0,80** |
| 6 | 0,57 | 0,200 | 904 | 1,46 | 0,72 |
| 7 | 0,60 | 0,180 | 830 | 1,54 | 0,73 |
| 8 | 0,61 | 0,166 | 761 | 1,61 | 0,54 |

**Désaccords entre indices.** Silhouette, CH et Dunn préfèrent k = 2 ; coude de l'inertie et
gap statistic k = 3 ; BIC (mélange gaussien) k = 6. Dans un nuage continu, les indices fondés
sur la séparation récompensent la coupure la plus grossière (actifs / inactifs), le R² croît
mécaniquement avec k, et le BIC (ellipses gaussiennes) découpe plus finement.

**Choix : k = 5**, parce que :
- **stabilité** : ARI bootstrap 0,80 (meilleur que k = 4, 6, 7, 8), Jaccard par segment
  0,86 / 0,89 / 0,93 / 0,73 / 0,69 (aucun segment ne se dissout) ;
- **silhouette** 0,205 avec seulement 1 à 4 % de clients à silhouette négative par segment ;
- **utilité métier** : le taux de réachat va de 34 % à 91 % selon le segment (global 57 %),
  homogénéité externe maximale (0,130) sur k = 2..8 ; k = 3 mélangerait les Champions (67 % du CA)
  avec les réguliers ; 5 segments = 5 actions commerciales distinctes.

### Module 5 – Segments, explications et actions

| # | Segment | Clients | CA | Réachat | Profil (médianes) | Action |
|---|---|---|---|---|---|---|
| 2 | **Champions** | 18 % | **67 %** | 91 % | 7 commandes, 2 791 £, dernier achat il y a 19 j | Programme VIP, gestionnaire dédié, avant-premières ; pas de remise systématique |
| 0 | **Clients réguliers** | 24 % | 14 % | 68 % | 3 commandes, 776 £, anciens (222 j) | Ventes croisées, fidélité à paliers pour les faire monter en Champions |
| 4 | **Nouveaux clients à gros panier** | 17 % | 12 % | 52 % | 1 commande, panier médian 505 £, récents | Contact commercial, offre volume / compte pro (revendeurs) |
| 3 | **Nouveaux petits acheteurs** | 18 % | 2 % | 43 % | 1 commande, 187 £, récents | Parcours de bienvenue, bon sur la 2e commande |
| 1 | **Clients endormis** | 23 % | 4 % | 34 % | dernier achat il y a 206 j, 1 commande | Réactivation peu coûteuse, puis arrêt des envois |

- **Arbre de substitution** (profondeur 4) : fidélité 84,8 % (83,5 % en validation croisée),
  règles traduites en phrases, ex. : *« Si Récence ≤ 145 jours et Montant > 1 774 £ et
  Fréquence ≥ 4 commandes, alors le client est classé Champions (496 clients, 94 % bien classés) »*.
- **Explication d'une affectation** (onglet 6 et API) : segment, distance au centre, segment
  suivant et marge, variables qui l'éloignent le plus du client moyen (z-scores, phrases),
  part de chaque variable dans la distance au centre, règle de l'arbre, et drapeau **client
  atypique** (distance > 95e centile de son segment ou une variable au-delà de 3 écarts-types).

### Module 6 – API FastAPI

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health`, `/model/info` | état, configuration, métriques, noms des axes |
| GET | `/segments`, `/segments/{id}` | nom, action, effectif, médianes, indices de profil, règles |
| POST | `/predict` | affecte un client décrit par ses 7 variables et explique l'affectation |
| POST | `/predict/batch` | idem pour une liste de clients |
| POST | `/predict/transactions` | construit le profil d'un nouveau client à partir de ses lignes de facture brutes puis l'affecte |
| POST | `/model/reload` | recharge le modèle après un nouvel enregistrement depuis l'application |

Exemple :
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"Recence\": 10, \"Frequence\": 12, \"Montant\": 8000, \"PanierMoyen\": 650, \"NbProduits\": 200, \"Anciennete\": 270, \"TauxRetour\": 2}"
```
→ `Champions`, distance 1,32, non atypique, règle : *Récence ≤ 145 jours et Montant > 1 774 £ et Fréquence ≥ 4 commandes*.

## 4. Limites
- Les silhouettes restent modestes (≈ 0,2) : les segments découpent un continuum de
  comportements plutôt que des groupes naturellement séparés ; c'est pourquoi la stabilité et
  l'utilité métier pèsent dans le choix.
- La fenêtre d'observation de 9 mois sert à créer une étiquette externe honnête ; pour la mise
  en production, on recalculerait le profil sur toute la période (option dans l'application).
- 90 % des clients sont britanniques : le pays n'est pas utilisé, il sert seulement à décrire.
