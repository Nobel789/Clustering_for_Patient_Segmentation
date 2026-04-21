# Patient Segmentation FastAPI

This API trains a **K-Means** model from `patient_segmentation/patient_dataset.csv` and exposes:

- `GET /health`
- `POST /predict_risk` → returns `Low`, `Medium`, or `High`

## 1) Install

```bash
cd patient_segmentation/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 3) Test endpoint

```bash
curl -X POST 'http://127.0.0.1:8000/predict_risk' \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 52,
    "blood_pressure": 140,
    "cholesterol": 220,
    "max_heart_rate": 145,
    "plasma_glucose": 180,
    "skin_thickness": 34,
    "insulin": 130,
    "bmi": 31.5,
    "diabetes_pedigree": 0.8,
    "hypertension": 1,
    "heart_disease": 0
  }'
```

Then open docs at `http://127.0.0.1:8000/docs`.
