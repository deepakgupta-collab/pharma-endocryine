"""
Endocrine Disruption Toxicity Predictor
Step 1-4: Data labeling, featurization, training, evaluation
Run: python train_model.py
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & Label ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("file.csv")

# Keep only rows with SMILES and IC50 in nM
df = df[
    df['Smiles'].notna() &
    df['Standard Value'].notna() &
    (df['Standard Units'] == 'nM')
].copy()

# Label: IC50 <= 1000 nM = Toxic (1), else Non-toxic (0)
# This threshold is standard for endocrine disruption studies
df['label'] = (df['Standard Value'] <= 1000).astype(int)

print(f"Total samples: {len(df)}")
print(f"  Toxic   (1): {df['label'].sum()}")
print(f"  Non-toxic (0): {(df['label'] == 0).sum()}")

# ── 2. SMILES → Morgan Fingerprint ───────────────────────────────────────────
print("\nConverting SMILES to fingerprints...")

def smiles_to_fp(smi, radius=2, nbits=2048):
    """Convert a SMILES string to a 2048-bit Morgan fingerprint."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return list(fp)

fps = df['Smiles'].apply(smiles_to_fp)
valid = fps.notna()

X = np.array(fps[valid].tolist())
y = df['label'][valid].values

print(f"Valid molecules: {len(X)} | Feature size: {X.shape[1]} bits")

# ── 3. Train / Test Split & Model ────────────────────────────────────────────
print("\nTraining Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Results ──────────────────────────────────")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-toxic', 'Toxic']))

# ── 5. Save Model ─────────────────────────────────────────────────────────────
joblib.dump(model, "toxicity_model.pkl")
print("Model saved → toxicity_model.pkl")
print("\nDone! Now run: python app.py  to start the API")
