"""
app.py  —  AutiSense Web Application
======================================
Autism Severity Assessment using the trained DREAM 2020 ML pipeline.

Model preference order:
  1. stacking_model_v7final.pkl  — Stacking Ensemble (top-3 models, ~98% LOOCV)
  2. best_model_v7final.pkl      — Best overall model saved by the pipeline
  3. best_model_v6ha.pkl         — Fallback from v6 high-accuracy run

Input features match exactly what the model was trained on (DREAM 2020 dataset):
  - ados_pre_comm, ados_pre_interact, ados_pre_play, ados_pre_stereo, ados_pre_scq
  - gender, age_months
  - All sensor features (gaze + skeleton) → imputed with training-set medians

Run:
    python app.py
Open:   http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))

# Prefer Stacking Ensemble (more robust, avoids near-perfect feature-label
# correlation of LR). Fall back if stacking pkl not yet generated.
MODEL_CANDIDATES = [
    os.path.join(_HERE, "stacking_model_v7final.pkl"),   # preferred
    os.path.join(_HERE, "best_model_v7final.pkl"),
    os.path.join(_HERE, "best_model_v6ha.pkl"),
    os.path.join(_HERE, "best_model_dream.pkl"),
]

model_data = None


def load_model():
    global model_data
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            try:
                model_data = joblib.load(path)
                name = model_data.get("model_name", "Unknown")
                acc  = model_data.get("loocv_accuracy", 0) * 100
                print(f"  [OK] Model: {name}  ({acc:.1f}% LOOCV)  <- {os.path.basename(path)}")
                return
            except Exception as e:
                print(f"  [SKIP] {os.path.basename(path)}: {e}")

    raise RuntimeError(
        "[ERROR] No model file found. Run ml_dream_v7_final.py first."
    )


load_model()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name=model_data.get("model_name", "ML Classifier"),
        model_acc=f"{model_data.get('loocv_accuracy', 0)*100:.1f}",
        threshold=model_data.get("ados_median_threshold", 14.0),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model_data is None:
        return jsonify(error="Model not loaded. Run ml_dream_v7_final.py first."), 500

    try:
        d = request.get_json(force=True)

        # ── Parse inputs — exactly mirroring DREAM 2020 feature names ─────────
        gender          = 0 if str(d.get("gender", "male")).lower() == "male" else 1
        age_months      = max(1.0, float(d.get("age_years", 5))) * 12.0
        ados_pre_comm   = float(d.get("ados_comm",     0))   # 0–8
        ados_pre_inter  = float(d.get("ados_interact",  0))  # 0–14
        ados_pre_play   = float(d.get("ados_play",     0))   # 0–5
        ados_pre_ster   = float(d.get("ados_stereo",   0))   # 0–8
        ados_pre_scq    = float(d.get("ados_scq",      0))   # 0–39

        # ── Build full feature row (NaN for all sensor features) ──────────────
        # The pipeline's SimpleImputer fills sensor NaNs with training medians.
        feat_cols = model_data["feature_cols"]
        feat_row  = {col: np.nan for col in feat_cols}
        feat_row.update({
            "gender":            gender,
            "age_months":        age_months,
            "ados_pre_comm":     ados_pre_comm,
            "ados_pre_interact": ados_pre_inter,
            "ados_pre_play":     ados_pre_play,
            "ados_pre_stereo":   ados_pre_ster,
            "ados_pre_scq":      ados_pre_scq,
        })

        X    = pd.DataFrame([feat_row])[feat_cols].values
        pred = int(model_data["model"].predict(X)[0])
        prob = model_data["model"].predict_proba(X)[0].tolist()

        p_low  = round(prob[0] * 100, 1)
        p_high = round(prob[1] * 100, 1)
        conf   = round(max(prob) * 100, 1)

        risk = "High" if p_high >= 65 else ("Moderate" if p_high >= 40 else "Low")

        total = round(ados_pre_comm + ados_pre_inter + ados_pre_play + ados_pre_ster, 1)
        thr   = float(model_data.get("ados_median_threshold", 14.0))

        # Per-domain percentage of max possible score (for breakdown bars)
        domain_pcts = {
            "Communication   (ados_pre_comm)":      round(ados_pre_comm  / 8  * 100),
            "Social Interact (ados_pre_interact)":  round(ados_pre_inter / 14 * 100),
            "Play            (ados_pre_play)":      round(ados_pre_play  / 5  * 100),
            "Stereotypy      (ados_pre_stereo)":    round(ados_pre_ster  / 8  * 100),
            "SCQ             (ados_pre_scq)":       round(ados_pre_scq   / 39 * 100),
        }

        return jsonify(
            prediction   = pred,
            severity     = "Higher Severity" if pred == 1 else "Lower Severity",
            confidence   = conf,
            prob_lower   = p_low,
            prob_higher  = p_high,
            risk_level   = risk,
            total_score  = total,
            threshold    = thr,
            domain_pcts  = domain_pcts,
            model_name   = model_data.get("model_name", "ML Classifier"),
            model_acc    = round(model_data.get("loocv_accuracy", 0) * 100, 1),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error=str(e)), 400


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AutiSense — DREAM 2020 Severity Assessment")
    print("  Open: http://localhost:5000")
    print("="*55)
    app.run(debug=True, host="0.0.0.0", port=5000)