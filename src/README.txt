# CPI-Triage-AI (Applications-ready, CRD-free)

Includes **WorkflowTemplates** for training and daily summary.

## How to use
1. Build & push Docker images:
   ```bash
   docker build -f docker/Dockerfile.train -t <your-registry>/<repo>:cpi-triage-ai-train .
   docker build -f docker/Dockerfile.serve -t <your-registry>/<repo>:cpi-triage-ai-serve .
   docker push <your-registry>/<repo>:cpi-triage-ai-train
   docker push <your-registry>/<repo>:cpi-triage-ai-serve
