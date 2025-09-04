from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os, io, joblib, boto3
from botocore.config import Config

app = FastAPI(title="CPI Simple Classifier (AI Core)")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")
VEC_PATH = os.getenv("VEC_PATH", "/app/model/vectorizer.pkl")
S3_MODEL_URI = os.getenv("S3_MODEL_URI")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

def _parse_s3_uri(uri: str):
    assert uri.startswith("s3://"), f"Not an S3 URI: {uri}"
    p = uri[5:]
    bucket, key = p.split("/", 1)
    return bucket, key

def download_s3_to_bytes(uri: str) -> bytes:
    bucket, key = _parse_s3_uri(uri)
    s3 = boto3.client("s3", config=Config(region_name=AWS_REGION))
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

def ensure_model_loaded():
    global model, vec
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model = joblib.load(MODEL_PATH)
        vec = joblib.load(VEC_PATH)
        return
    if S3_MODEL_URI:
        print(f"Loading model from {S3_MODEL_URI}")
        model_bytes = download_s3_to_bytes(S3_MODEL_URI)
        model = joblib.load(io.BytesIO(model_bytes))
        vec_uri = S3_MODEL_URI.rsplit("/", 1)[0] + "/vectorizer.pkl"
        vec_bytes = download_s3_to_bytes(vec_uri)
        vec = joblib.load(io.BytesIO(vec_bytes))
        return
    raise RuntimeError("Model not found locally and S3_MODEL_URI not provided.")

ensure_model_loaded()

class CPILog(BaseModel):
    ARTIFACT_NAME: Optional[str] = ""
    ORIGIN_COMPONENT_NAME: Optional[str] = ""
    LOG_LEVEL: Optional[str] = ""

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(record: CPILog) -> Dict[str, Any]:
    X = vec.transform([record.dict()])
    pred = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf = float(max(proba))
    else:
        conf = None
    return {"prediction": pred, "confidence": conf}
