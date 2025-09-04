import os, io, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import boto3
from botocore.config import Config

DATA_PATH = os.getenv("DATA_PATH", "/app/data/cpi_logs.csv")
S3_DATA_URI = os.getenv("S3_DATA_URI")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))
VEC_PATH = os.getenv("VEC_PATH", os.path.join(MODEL_DIR, "vectorizer.pkl"))
S3_MODEL_URI = os.getenv("S3_MODEL_URI")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

os.makedirs(MODEL_DIR, exist_ok=True)

def _parse_s3_uri(uri: str):
    assert uri.startswith("s3://"), f"Not an S3 URI: {uri}"
    p = uri[5:]
    bucket, key = p.split("/", 1)
    return bucket, key

def read_csv_from_s3(uri: str) -> pd.DataFrame:
    bucket, key = _parse_s3_uri(uri)
    s3 = boto3.client("s3", config=Config(region_name=AWS_REGION))
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def upload_file_to_s3(local_path: str, uri: str):
    bucket, key = _parse_s3_uri(uri)
    s3 = boto3.client("s3", config=Config(region_name=AWS_REGION))
    s3.upload_file(local_path, bucket, key)
    print(f"Uploaded {local_path} → {uri}")

# Load data
if S3_DATA_URI:
    print(f"Reading training data from {S3_DATA_URI}")
    df = read_csv_from_s3(S3_DATA_URI)
else:
    print(f"Reading training data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

target_col = os.getenv("TARGET_COLUMN", "CUSTOM_STATUS")
if target_col not in df.columns:
    raise ValueError(f"TARGET_COLUMN '{target_col}' missing in dataset columns: {list(df.columns)}")

feat_cols = [c for c in ["ARTIFACT_NAME","ORIGIN_COMPONENT_NAME","LOG_LEVEL"] if c in df.columns]
if not feat_cols:
    raise ValueError("No feature columns found. Expected at least one of: ARTIFACT_NAME, ORIGIN_COMPONENT_NAME, LOG_LEVEL")

X_dict = df[feat_cols].fillna("").to_dict(orient="records")
y = df[target_col].astype(str)

vec = DictVectorizer(sparse=True)
X = vec.fit_transform(X_dict)

strat = y if y.nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(os.getenv("TEST_SIZE","0.2")), random_state=42, stratify=strat)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", round(acc, 4))
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, MODEL_PATH)
joblib.dump(vec, VEC_PATH)
print(f"Saved model → {MODEL_PATH}")
print(f"Saved vectorizer → {VEC_PATH}")

if S3_MODEL_URI:
    upload_file_to_s3(MODEL_PATH, S3_MODEL_URI)
    vec_uri = S3_MODEL_URI.rsplit("/", 1)[0] + "/vectorizer.pkl"
    upload_file_to_s3(VEC_PATH, vec_uri)
