"""
Toxicity Prediction API
Step 5: Flask backend that receives SMILES and returns toxic/non-toxic

Install: pip install flask rdkit scikit-learn joblib
Run:     python app.py
Test:    curl -X POST http://localhost:5000/predict \
              -H "Content-Type: application/json" \
              -d '{"smiles": "CCO"}'
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# Load model once at startup
model = joblib.load("toxicity_model.pkl")

def smiles_to_fp(smi, radius=2, nbits=2048):
    """Convert SMILES to Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp).reshape(1, -1)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives: {"smiles": "CCO"}
    Returns:  {"smiles": "CCO", "toxic": false, "probability": 0.12, "label": "Non-toxic"}
    """
    data = request.get_json()

    if not data or "smiles" not in data:
        return jsonify({"error": "Please send JSON with a 'smiles' field"}), 400

    smiles = data["smiles"].strip()
    fp = smiles_to_fp(smiles)

    if fp is None:
        return jsonify({"error": f"Invalid SMILES string: {smiles}"}), 400

    prob = model.predict_proba(fp)[0][1]  # probability of being toxic
    is_toxic = bool(prob >= 0.5)

    return jsonify({
        "smiles": smiles,
        "toxic": is_toxic,
        "probability": round(float(prob), 4),
        "label": "Toxic" if is_toxic else "Non-toxic"
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})

if __name__ == "__main__":
    print("Starting Toxicity Prediction API on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /predict  body: {\"smiles\": \"your_smiles_here\"}")
    app.run(debug=True, port=5000)
