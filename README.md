# CPI‑Triage‑AI (Applications-ready)

This repo is structured so **AI Launchpad → Applications → Import** can detect:
- Scenario **CPI-Triage-AI** with two **Executables** (training + daily summary) from Argo **WorkflowTemplates**
- One **Deployment** (AI Core CRD) exposing `/predict`

## Folders
- `applications/`
  - `cpi-triage-ai-train.yaml` – WorkflowTemplate that trains and outputs a `cpimodel` artifact
  - `cpi-triage-ai-daily-summary.yaml` – WorkflowTemplate to batch predict & write `summaryout`
  - `cpi-triage-ai-service.yaml` – AI Core Deployment (serving)
- `docker/`
  - `Dockerfile.train` – build image for train + summary
  - `Dockerfile.serve` – build image for serving
- `src/`
  - `train_cpi_btpcore.py` – reads dataset (S3 or artifact), trains model
  - `serve_cpi_btpcore.py` – loads model (artifact or S3) and serves FastAPI
  - `summary_job.py` – batch predict and write summary.json / bucket_summary.csv
  - `requirements.txt`

## Replace placeholders
- `<your-docker-secret>` – image pull secret name in your AI Core runtime namespace
- `<your-registry>/<repo>` – container registry paths for the images you build

## Build & push images
```bash
docker build -f docker/Dockerfile.train -t <your-registry>/<repo>:cpi-triage-ai-train .
docker build -f docker/Dockerfile.serve -t <your-registry>/<repo>:cpi-triage-ai-serve .
docker push <your-registry>/<repo>:cpi-triage-ai-train
docker push <your-registry>/<repo>:cpi-triage-ai-serve
```

## Import into Applications
In **AI Launchpad → Applications → Add → YAML**, paste each YAML from `applications/`. Or push this repo to a Git provider and configure Applications to watch the branch.

### Training run
- Executable: from `cpi-triage-ai-train`
- Parameters:
  - `S3_DATA_URI` (optional) e.g. `s3://bucket/path/cpi_logs.csv`
  - `TARGET_COLUMN` default `CUSTOM_STATUS`
- Input Artifact (optional): attach your CSV as `cpidataset` (will be placed at `/app/data/`)
- Output Artifact: `cpimodel` at `/app/model/`

### Daily summary
- Executable: `cpi-triage-ai-daily-summary`
- Inputs: `cpidataset` (optional), **cpimodel** (required)
- Output: `summaryout` (JSON/CSV)

### Serving
- Deployment: `cpi-triage-ai-service`
- Attach model artifact **cpimodel** from training OR set `S3_MODEL_URI` env manually.
- Test:
```bash
curl -s https://<endpoint>/health
curl -s -X POST https://<endpoint>/predict -H 'Content-Type: application/json'   -d '{"ARTIFACT_NAME":"InvoiceProcessing","ORIGIN_COMPONENT_NAME":"HTTP","LOG_LEVEL":"ERROR"}'
```

## Notes
- All templates use **scenario id**: `cpi-triage-ai`
- For S3, provide `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` to the runtime via secret/variables.
