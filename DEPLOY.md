Deployment instructions
=======================

This document describes easy ways to deploy the Streamlit insurance claims fraud detection app.

1) Streamlit Cloud (fastest)

- Push your repository to GitHub (public or grant Streamlit access).
- Go to https://share.streamlit.io and sign in.
- Create a new app and point it to the `app.py` file in the repository.
- Streamlit will install packages from `requirements.txt` and run the app.

Notes: If your data file is large you may want to store it externally (S3, GCS, etc.) and set the `DATA_PATH` environment variable or fetch it from a remote location.

2) Heroku

- Ensure `Procfile` exists (already included):

  `web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

- Ensure `.streamlit/config.toml` exists (included).
- Create a Heroku app and push:

```bash
heroku login
heroku create your-app-name
git push heroku main
heroku config:set DATA_PATH="csv/fraud_insurance_claims.csv"
heroku ps:scale web=1
```

3) Docker (portable, good for cloud providers)

- Build the image locally:

```bash
docker build -t insurance-fraud-app .
```

- Run the container:

```bash
docker run -p 8501:8501 --env DATA_PATH="csv/fraud_insurance_claims.csv" insurance-fraud-app
```

4) Cloud providers (AWS/GCP/Azure)

- Push the `Dockerfile` to a registry (Docker Hub, ECR, GCR, ACR) and deploy to the provider's container service (Cloud Run, ECS, App Service, etc.).

Local testing
-------------
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run locally:

```bash
streamlit run app.py
```

Environment and data
--------------------
- The app uses the `DATA_PATH` environment variable if present; otherwise it falls back to `csv/fraud_insurance_claims.csv`.
- If you keep data out of the repo, set `DATA_PATH` to a remote URL or to a mounted/accessible path in the deployed container.
