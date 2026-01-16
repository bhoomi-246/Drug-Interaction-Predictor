# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from scipy.special import softmax

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
INTERACTION_MODEL_PATH = os.path.join(MODEL_DIR, "interaction_model.pkl")
SEVERITY_MODEL_PATH = os.path.join(MODEL_DIR, "severity_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Load models
with open(INTERACTION_MODEL_PATH, "rb") as f:
    interaction_model = pickle.load(f)
with open(SEVERITY_MODEL_PATH, "rb") as f:
    severity_model = pickle.load(f)
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

interaction_encoder = encoders["interaction_encoder"]
severity_encoder = encoders["severity_encoder"]

app = Flask(__name__)
CORS(app)

def safe_predict_with_scores(pipeline, texts):
    """Return predicted label index and softmaxed scores for decision_function output.
    If classifier doesn't implement decision_function, falls back to predict (binary/no scores).
    """
    try:
        df = pipeline.decision_function(texts)
        # multi-class -> shape (n_samples, n_classes) ; binary -> (n_samples,)
        if df.ndim == 1:
            # binary case: convert to two-class scores
            scores = np.vstack([-df, df]).T
        else:
            scores = df
        probs = softmax(scores, axis=1)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    except Exception:
        # fallback
        preds = pipeline.predict(texts)
        # no score info; create uniform probability
        probs = np.ones((len(preds), len(np.unique(preds)))) / len(np.unique(preds))
        return preds, probs

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    drug1 = data.get('drug1', '').strip()
    drug2 = data.get('drug2', '').strip()
    if not drug1 or not drug2:
        return jsonify({'error': 'Provide both drug1 and drug2'}), 400

    pair = f"{drug1} {drug2}"
    # Interaction
    int_pred_idx, int_probs = safe_predict_with_scores(interaction_model, [pair])
    int_label = interaction_encoder.inverse_transform(int_pred_idx)[0]
    int_scores = int_probs[0].tolist()
    int_classes = interaction_encoder.inverse_transform(np.arange(len(int_scores))).tolist()

    # Severity
    sev_pred_idx, sev_probs = safe_predict_with_scores(severity_model, [pair])
    sev_label = severity_encoder.inverse_transform(sev_pred_idx)[0]
    sev_scores = sev_probs[0].tolist()
    sev_classes = severity_encoder.inverse_transform(np.arange(len(sev_scores))).tolist()

    return jsonify({
        'drug1': drug1,
        'drug2': drug2,
        'interaction': {
            'label': int_label,
            'classes': int_classes,
            'scores': int_scores
        },
        'severity': {
            'label': sev_label,
            'classes': sev_classes,
            'scores': sev_scores
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)