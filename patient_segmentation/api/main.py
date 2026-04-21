from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="Patient Segmentation API",
    description="Predict patient risk cluster (Low / Medium / High) from clinical features.",
    version="1.0.0",
)

DATA_PATH = Path(__file__).resolve().parents[1] / "patient_dataset.csv"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "age",
    "blood_pressure",
    "cholesterol",
    "max_heart_rate",
    "plasma_glucose",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "hypertension",
    "heart_disease",
]

RISK_RELEVANT = ["plasma_glucose", "bmi", "blood_pressure", "cholesterol", "age"]


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=0, le=120)
    blood_pressure: float = Field(..., ge=0)
    cholesterol: float = Field(..., ge=0)
    max_heart_rate: float = Field(..., ge=0)
    plasma_glucose: float = Field(..., ge=0)
    skin_thickness: float = Field(..., ge=0)
    insulin: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)
    diabetes_pedigree: float = Field(..., ge=0)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    cluster_id: int
    risk_cluster: Literal["Low", "Medium", "High"]
    distances: Dict[str, float]


class ModelBundle:
    def __init__(self) -> None:
        self.pipeline, self.risk_mapping = self._train()

    @staticmethod
    def _train() -> tuple[Pipeline, Dict[int, str]]:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset was not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)
        missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")

        X = df[FEATURE_COLUMNS].copy()

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)),
            ]
        )
        pipeline.fit(X)

        labels = pipeline.named_steps["kmeans"].labels_
        scored = df[RISK_RELEVANT].copy()
        scored["cluster"] = labels
        cluster_scores = scored.groupby("cluster").mean().sum(axis=1).sort_values()

        ordered_clusters = cluster_scores.index.tolist()
        risk_mapping = {
            ordered_clusters[0]: "Low",
            ordered_clusters[1]: "Medium",
            ordered_clusters[2]: "High",
        }
        return pipeline, risk_mapping


model_bundle = ModelBundle()


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict_risk", response_model=PredictionResponse)
def predict_risk(payload: PatientFeatures) -> PredictionResponse:
    try:
        input_df = pd.DataFrame([payload.model_dump()])[FEATURE_COLUMNS]
        cluster_id = int(model_bundle.pipeline.predict(input_df)[0])

        transformed = model_bundle.pipeline[:-1].transform(input_df)
        dists = model_bundle.pipeline.named_steps["kmeans"].transform(transformed)[0]
        named_distances = {f"cluster_{i}": float(round(dist, 4)) for i, dist in enumerate(dists)}

        return PredictionResponse(
            cluster_id=cluster_id,
            risk_cluster=model_bundle.risk_mapping[cluster_id],
            distances=named_distances,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
