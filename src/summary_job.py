import os, io, json, joblib, pandas as pd, numpy as np, requests, boto3
from botocore.config import Config
from collections import Counter

DATA_PATH = os.getenv("DATA_PATH", "/app/data/cpi_logs.csv")
S3_DATA_URI = os.getenv("S3_DATA_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")
VEC_PATH = os.getenv("VEC_PATH", "/app/model/vectorizer.pkl")
OUT_DIR = os.getenv("OUT_DIR", "/app/out")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

os.makedirs(OUT_DIR, exist_ok=True)

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

# Load data
if S3_DATA_URI:
    df = read_csv_from_s3(S3_DATA_URI)
else:
    df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)
vec = joblib.load(VEC_PATH)

feat_cols = [c for c in ["ARTIFACT_NAME","ORIGIN_COMPONENT_NAME","LOG_LEVEL"] if c in df.columns]
X = vec.transform(df[feat_cols].fillna("").to_dict(orient="records"))
preds = model.predict(X)

df["_pred"] = preds
dist = df["_pred"].value_counts().to_dict()
avg_by_loglvl = df.groupby("LOG_LEVEL")["_pred"].value_counts().unstack(fill_value=0).to_dict()

summary = {
    "total_records": int(len(df)),
    "predicted_distribution": dist,
    "by_log_level": {k: {kk:int(vv) for kk,vv in v.items()} for k,v in avg_by_loglvl.items()}
}

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

pd.DataFrame([{"bucket": k, "count": v} for k, v in dist.items()])  .to_csv(os.path.join(OUT_DIR, "bucket_summary.csv"), index=False)

headline = f"CPI Daily Summary: {summary['total_records']} logs | " + ", ".join(f"{k}:{v}" for k,v in dist.items())
def post(url: str, payload: dict):
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print("Notify error:", e)

if SLACK_WEBHOOK_URL:
    post(SLACK_WEBHOOK_URL, {"text": headline + (f" | Dashboard: {DASHBOARD_URL}" if DASHBOARD_URL else "")})

if TEAMS_WEBHOOK_URL:
    card = {
      "@type": "MessageCard",
      "@context": "http://schema.org/extensions",
      "summary": "CPI Daily Summary",
      "themeColor": "0076D7",
      "sections": [{
        "activityTitle": "CPI Daily Summary",
        "facts": [{"name":"Total Logs","value":str(summary['total_records'])}] +                  [{"name": f"Bucket {k}", "value": str(v)} for k,v in dist.items()],
        "markdown": True
      }]
    }
    if DASHBOARD_URL:
        card["potentialAction"] = [{
          "@type": "OpenUri",
          "name": "Open Dashboard",
          "targets": [{"os":"default","uri":DASHBOARD_URL}]
        }]
    post(TEAMS_WEBHOOK_URL, card)

print("Wrote summary to", OUT_DIR)
