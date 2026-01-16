import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('/mnt/data/DDI_data.csv')

# ==== Severity Mapping ====
def map_severity(text):
    t = text.lower()
    if "risk or severity" in t:
        return "severe"
    if "anticoagulant" in t or "metabolism" in t:
        return "moderate"
    return "mild"

df["severity"] = df["interaction_type"].apply(map_severity)

# Combine drug names into one input text
df["drug_pair"] = df["drug1_name"] + " " + df["drug2_name"]

# ==== Label Encoders ====
le_interaction = LabelEncoder()
df["interaction_label"] = le_interaction.fit_transform(df["interaction_type"])

le_severity = LabelEncoder()
df["severity_label"] = le_severity.fit_transform(df["severity"])

# ==== Train INTERACTION Model ====
interaction_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])

interaction_model.fit(df["drug_pair"], df["interaction_label"])

# ==== Train SEVERITY Model ====
severity_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])

severity_model.fit(df["drug_pair"], df["severity_label"])

# ==== Save the Models ====
interaction_path = "/mnt/data/interaction_model.pkl"
severity_path = "/mnt/data/severity_model.pkl"
encoder_path = "/mnt/data/label_encoders.pkl"

with open(interaction_path, "wb") as f:
    pickle.dump(interaction_model, f)

with open(severity_path, "wb") as f:
    pickle.dump(severity_model, f)

with open(encoder_path, "wb") as f:
    pickle.dump({
        "interaction_encoder": le_interaction,
        "severity_encoder": le_severity
    }, f)

interaction_path, severity_path, encoder_path
