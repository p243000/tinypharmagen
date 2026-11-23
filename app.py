#!/usr/bin/env python3
"""TinyPharmaGen local demo server.

Serves a monochrome Three.js frontend and an XGBoost inference API.

Run:  python app.py            ->  http://127.0.0.1:8000
"""
import io
import json
import os
import re
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from features import pair_features

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / "figures"
OUTDIR = HERE / "outputs"
HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
AA = set("ACDEFGHIKLMNPQRSTVWY")
AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
FP_BITS, N_DESC = 1024, 7

with open(HERE / "meta.json", encoding="utf-8") as f:
    META = json.load(f)
KINASES = pd.read_csv(HERE / "kinases.csv").fillna("")
MODEL = xgb.XGBRegressor()
MODEL.load_model(str(HERE / "model.json"))


# --------------------------------------------------------------------------- #
#  Feature names (must mirror features.py order)
# --------------------------------------------------------------------------- #
def _feature_names():
    names = [f"fp_bit_{i}" for i in range(FP_BITS)]
    names += ["MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Rings"]
    names += [f"AAC_{a}" for a in AA_ORDER]
    names += [f"DPC_{a}{b}" for a in AA_ORDER for b in AA_ORDER]
    return names


FEATURE_NAMES = _feature_names()


def _build_model_summary():
    """Model statistics: leaderboard, learning curve, feature importances."""
    summary = {"results": [], "learning_curve": [], "figures": [],
               "importance": {"groups": [], "top": []}}

    # ---- results table ---------------------------------------------------- #
    res_path = OUTDIR / f"results_{META['dataset'].lower()}.csv"
    if res_path.exists():
        df = pd.read_csv(res_path)
        keep = ["split", "model", "MSE", "RMSE", "R2", "Pearson", "Spearman", "CI", "secs"]
        df = df[[c for c in keep if c in df.columns]].round(4)
        summary["results"] = df.to_dict(orient="records")

    # ---- learning curve --------------------------------------------------- #
    lc_path = OUTDIR / f"learning_curve_{META['dataset'].lower()}.csv"
    if lc_path.exists():
        lc = pd.read_csv(lc_path)
        g = lc.groupby("n_drugs").agg(CI=("CI", "mean"), CI_std=("CI", "std"),
                                      MSE=("MSE", "mean")).reset_index()
        summary["learning_curve"] = g.round(4).to_dict(orient="records")

    # ---- feature importance (gain) --------------------------------------- #
    try:
        scores = MODEL.get_booster().get_score(importance_type="gain")
        imp = np.zeros(len(FEATURE_NAMES), dtype=float)
        for k, v in scores.items():
            m = re.fullmatch(r"f(\d+)", str(k))
            if m and int(m.group(1)) < len(imp):
                imp[int(m.group(1))] = v
        if imp.sum() > 0:
            groups = OrderedDict([
                ("Drug · Morgan fingerprint", float(imp[:FP_BITS].sum())),
                ("Drug · descriptors", float(imp[FP_BITS:FP_BITS + N_DESC].sum())),
                ("Protein · amino-acid composition",
                 float(imp[FP_BITS + N_DESC:FP_BITS + N_DESC + 20].sum())),
                ("Protein · dipeptide composition",
                 float(imp[FP_BITS + N_DESC + 20:].sum())),
            ])
            total = sum(groups.values()) or 1.0
            summary["importance"]["groups"] = [
                {"name": k, "share": round(v / total, 4)} for k, v in groups.items()
            ]

            order = np.argsort(-imp)[:12]
            summary["importance"]["top"] = [
                {"name": FEATURE_NAMES[i], "value": float(imp[i])} for i in order
            ]

            pd.DataFrame({
                "feature": FEATURE_NAMES,
                "group": (["Drug · Morgan fingerprint"] * FP_BITS
                          + ["Drug · descriptors"] * N_DESC
                          + ["Protein · amino-acid composition"] * 20
                          + ["Protein · dipeptide composition"] * 400),
                "gain": imp,
                "share": imp / (imp.sum() or 1.0),
            }).to_csv(OUTDIR / "feature_importance.csv", index=False)
    except Exception as exc:  # pragma: no cover - defensive
        print("feature importance unavailable:", repr(exc))

    # ---- report figures ---------------------------------------------------- #
    if FIGDIR.exists():
        for p in sorted(FIGDIR.glob("*.png")):
            summary["figures"].append({
                "file": p.name,
                "title": _figure_title(p.stem),
            })
    return summary


def _figure_title(stem):
    return {
        "eda": "Exploratory data analysis",
        "ci_comparison": "Concordance Index by model",
        "pred_vs_actual": "Predicted vs actual affinity",
        "learning_curve": "Learning curve (cold-drug)",
        "feature_importance": "XGBoost feature importance",
    }.get(stem, stem.replace("_", " ").capitalize())


MODEL_SUMMARY = _build_model_summary()


# --------------------------------------------------------------------------- #
#  3D conformer extraction (ball-and-stick payload for Three.js)
# --------------------------------------------------------------------------- #
def molecule_3d(smiles):
    """Embed a 3D conformer and return atoms + bonds for the viewer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.useSmallRingTorsions = True
    if AllChem.EmbedMolecule(molh, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(molh, params) != 0:
            return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(molh):
            AllChem.MMFFOptimizeMolecule(molh, maxIters=600)
        else:
            AllChem.UFFOptimizeMolecule(molh, maxIters=600)
    except Exception:
        pass

    molh = Chem.RemoveHs(molh)  # clean ball-and-stick, implicit hydrogens
    conf = molh.GetConformer()
    coords = conf.GetPositions()
    coords = coords - coords.mean(axis=0)  # centre on origin

    atoms = [
        {"element": a.GetSymbol(),
         "x": round(float(coords[i, 0]), 4),
         "y": round(float(coords[i, 1]), 4),
         "z": round(float(coords[i, 2]), 4)}
        for i, a in enumerate(molh.GetAtoms())
    ]
    bonds = [
        {"a": b.GetBeginAtomIdx(), "b": b.GetEndAtomIdx(),
         "order": int(b.GetBondTypeAsDouble()) if b.GetBondTypeAsDouble() in (1, 2, 3) else 1,
         "aromatic": bool(b.GetIsAromatic())}
        for b in molh.GetBonds()
    ]
    return {"atoms": atoms, "bonds": bonds}


# --------------------------------------------------------------------------- #
#  HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "TinyPharmaGen/1.0"

    def _send(self, status, body, content_type="application/json; charset=utf-8", cache="no-store"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, payload, cache="no-store"):
        return self._send(status, json.dumps(payload), cache=cache)

    # ---------------- GET ---------------- #
    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            file = HERE / "index.html"
            if not file.exists():
                return self._send(404, "index.html not found", "text/plain")
            return self._send(200, file.read_bytes(), "text/html; charset=utf-8")

        if path == "/api/meta":
            return self._json(200, dict(
                META,
                features={
                    "drug": FP_BITS + N_DESC,
                    "protein": 20 + 400,
                    "pair": FP_BITS + N_DESC + 420,
                },
            ))

        if path == "/api/model":
            return self._json(200, MODEL_SUMMARY)

        if path == "/api/kinases":
            rows = [{"gene": str(r.gene), "sequence": str(r.sequence)}
                    for r in KINASES.itertuples(index=False)]
            return self._json(200, rows)

        if path == "/api/health":
            return self._json(200, {"status": "ok", "dataset": META["dataset"]})

        if path.startswith("/figures/"):
            name = unquote(path[len("/figures/"):])
            target = (FIGDIR / name).resolve()
            if target.is_file() and target.parent == FIGDIR.resolve() and target.suffix == ".png":
                return self._send(200, target.read_bytes(), "image/png", cache="public, max-age=3600")
            return self._send(404, "figure not found", "text/plain")

        if path == "/favicon.ico":
            return self._favicon()

        return self._json(404, {"error": "Not found"})

    # ---------------- POST ---------------- #
    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self._json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(length))

            smiles = str(data.get("smiles", "")).strip()
            seq = "".join(str(data.get("sequence", "")).upper().split())

            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                return self._json(400, {"error": "Invalid or empty SMILES. Check the molecular notation."})
            if len(seq) < 20 or any(c not in AA for c in seq):
                return self._json(400, {
                    "error": ("Protein sequence must contain at least 20 residues "
                              "and only the 20 standard amino-acid letters.")})

            canonical = Chem.MolToSmiles(mol)
            score = float(MODEL.predict(pair_features(canonical, seq))[0])
            label = ("Strong binder" if score >= META["strong"] else
                     "Moderate binder" if score >= META["moderate"] else
                     "Weak / non-binder")

            geometry = molecule_3d(canonical)
            image = None
            if geometry is None:
                buf = io.BytesIO()
                Draw.MolToImage(mol, size=(640, 460)).convert("L").save(buf, format="PNG")
                image = buf.getvalue().hex()

            result = {
                "score": round(score, 4),
                "label": label,
                "target_name": META["target_name"],
                "dataset": META["dataset"],
                "canonical_smiles": canonical,
                "cold_ci": META.get("cold_ci"),
                "thresholds": {"strong": META["strong"], "moderate": META["moderate"]},
                "geometry": geometry,
            }
            if image:
                result["molecule_png_hex"] = image
            return self._json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover
            print("Prediction error:", repr(exc))
            return self._json(500, {"error": "Prediction failed. Check model.json, features.py and dependencies."})

    def _favicon(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<rect width="64" height="64" rx="14" fill="#ffffff" stroke="#0a0a0a" stroke-width="3"/>'
            '<text x="32" y="43" font-family="monospace" font-size="30" text-anchor="middle" '
            'fill="#0a0a0a">&#8997;</text></svg>'
        )
        return self._send(200, svg.encode(), "image/svg+xml", cache="public, max-age=86400")

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


if __name__ == "__main__":
    print(f"\n  TinyPharmaGen demo  ->  http://{HOST}:{PORT}")
    print(f"  Dataset: {META['dataset']} | {META['n_drugs']} drugs | {META['n_kinases']} kinases"
          f" | results rows: {len(MODEL_SUMMARY['results'])}"
          f" | figures: {len(MODEL_SUMMARY['figures'])}")
    print("  Press Ctrl+C to stop.\n")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
