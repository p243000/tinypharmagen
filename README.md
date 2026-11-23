# TinyPharmaGen — Drug–Kinase Binding Affinity Prediction

Machine-learning system that predicts how strongly a small molecule binds to a kinase
target. Built for the **Artificial Intelligence** course project (BSE-5A).

- **Input:** drug SMILES + kinase amino-acid sequence
- **Output:** KIBA binding score (higher = stronger binding), binder class, interactive 3D conformer
- **Model:** XGBoost regressor on 1,451 engineered features
- **Frontend:** monochrome, Three.js-powered molecular workbench (ligand + protein 3D views)
- **Repository:** <https://github.com/p243000/tinypharmagen>

## Demo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:8000
```

Options:

| URL | Behaviour |
|---|---|
| `/` | Workbench |
| `/?demo=1` | Auto-runs a prediction (imatinib × first kinase) |
| `/?demo=1&static=1` | Same, with animations disabled (screenshots) |
| `/api/health` | Health check |
| `/api/meta` | Dataset + threshold metadata |
| `/api/model` | Leaderboard, learning curve, feature importances |
| `/api/predict` | `POST {smiles, sequence}` → score + 3D geometry |
| `/api/structure?gene=ACC` | AlphaFold Cα backbone trace + pLDDT (cached) |

**Demo video:** `demo/io.mp4` — https://github.com/p243000/tinypharmagen/blob/main/demo/io.mp4

## Results (KIBA, held-out test sets)

| Split | Model | MSE ↓ | R² ↑ | Pearson ↑ | Spearman ↑ | CI ↑ |
|---|---|---|---|---|---|---|
| Random | Random Forest | 0.322 | 0.540 | 0.751 | 0.756 | 0.805 |
| Random | XGBoost | 0.216 | 0.691 | 0.836 | 0.822 | 0.840 |
| Random | DeepDTA-lite CNN | **0.199** | **0.716** | **0.849** | **0.839** | **0.854** |
| Cold-drug | Random Forest | 0.382 | 0.473 | 0.699 | 0.716 | **0.787** |
| Cold-drug | **XGBoost** | **0.360** | **0.502** | **0.710** | 0.708 | 0.783 |
| Cold-drug | CNN | 0.421 | 0.419 | 0.657 | 0.656 | 0.760 |

CI = Concordance Index (primary metric). The **cold-drug split** holds out entire
molecules and is the realistic estimate for new compounds. XGBoost was selected:
tied with Random Forest on ranking quality, best cold-drug error metrics, fastest to
train and smallest to deploy.

## Features

| Block | Size | Description |
|---|---|---|
| Drug | 1,031 | 1024-bit Morgan fingerprint (radius 2) + 7 physicochemical descriptors |
| Protein | 420 | Amino-acid composition (20) + dipeptide composition (400) |
| Pair | 1,451 | Concatenated drug + protein vector |

## Repository layout

```
app.py               # HTTP server + inference API + 3D conformer generation
features.py          # Shared featurisation (training AND serving)
index.html           # Monochrome Three.js workbench with model visualisations
model.json           # Trained XGBoost model (10.5 MB)
kinases.csv          # 229 KIBA kinases (gene, sequence)
meta.json            # Dataset, thresholds, reported metrics
figures/             # Training figures + app screenshot
outputs/             # Results tables, learning curve, feature importance
notebook/            # Full training / EDA / comparison notebook
report/              # Report prompt and generated report PDF
docs/                # Assignment brief
```

## Dataset

KIBA and DAVIS benchmarks, downloaded from the public DeepDTA repository:
<https://github.com/hkmztrk/DeepDTA/tree/master/data>

- KIBA: 2,111 drugs × 229 kinases, 118,254 measured pairs
- DAVIS: 68 drugs × 442 kinases, 30,056 measured pairs

## Reproducing the training

```bash
pip install -r requirements-dev.txt
jupyter lab notebook/p243000_p243012_p243032_bse5a_ai_project.ipynb
```

Set `DATASET = "KIBA"` and `QUICK_RUN = False`, then run all cells. The notebook writes
`figures/` and `outputs/` directly.

## Team

| Name | Roll number |
|---|---|
| Vinod Kumar | 24P3012 |
| Pir Ahmed Shah | 24P3000 |
| Muhammad Maaz | 24P3032 |

Course: Artificial Intelligence — BSE-5A

## References

Davis et al., *Nat. Biotechnol.* 2011 · Tang et al., *J. Chem. Inf. Model.* 2014 (KIBA) ·
Öztürk et al., *Bioinformatics* 2018 (DeepDTA) · Breiman 2001 (Random Forest) ·
Chen & Guestrin, KDD 2016 (XGBoost) · Rogers & Hahn 2010 (Morgan/ECFP) ·
RDKit open-source cheminformatics toolkit.
