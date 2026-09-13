"""FastAPI REST API service for Electrical Grid Stability Classification."""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from config.config import BEST_MODEL_PATH
from src.prediction import PredictionError, load_metadata, load_model, predict_stability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grid_api")

app = FastAPI(
    title="Electrical Grid Stability Classification API",
    description="Production REST API for real-time binary classification of decentralized electrical grid stability.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


class GridStabilityInput(BaseModel):
    """Input payload representing electrical and dynamic grid parameters."""
    tau1: float = Field(..., description="Reaction time of participant 1 (generator)", ge=0.0, le=20.0, examples=[2.959])
    tau2: float = Field(..., description="Reaction time of participant 2 (consumer 1)", ge=0.0, le=20.0, examples=[3.080])
    tau3: float = Field(..., description="Reaction time of participant 3 (consumer 2)", ge=0.0, le=20.0, examples=[8.381])
    tau4: float = Field(..., description="Reaction time of participant 4 (consumer 3)", ge=0.0, le=20.0, examples=[9.781])

    p1: float = Field(..., description="Nominal power produced by generator node 1", ge=-15.0, le=15.0, examples=[3.763])
    p2: float = Field(..., description="Nominal power consumed by node 2 (negative)", ge=-15.0, le=15.0, examples=[-1.527])
    p3: float = Field(..., description="Nominal power consumed by node 3 (negative)", ge=-15.0, le=15.0, examples=[-1.390])
    p4: float = Field(..., description="Nominal power consumed by node 4 (negative)", ge=-15.0, le=15.0, examples=[-0.845])

    g1: float = Field(..., description="Price elasticity coefficient of node 1", ge=0.0, le=5.0, examples=[0.562])
    g2: float = Field(..., description="Price elasticity coefficient of node 2", ge=0.0, le=5.0, examples=[0.413])
    g3: float = Field(..., description="Price elasticity coefficient of node 3", ge=0.0, le=5.0, examples=[0.778])
    g4: float = Field(..., description="Price elasticity coefficient of node 4", ge=0.0, le=5.0, examples=[0.958])

    model_config = {
        "json_schema_extra": {
            "example": {
                "tau1": 2.959,
                "tau2": 3.080,
                "tau3": 8.381,
                "tau4": 9.781,
                "p1": 3.763,
                "p2": -1.527,
                "p3": -1.390,
                "p4": -0.845,
                "g1": 0.562,
                "g2": 0.413,
                "g3": 0.778,
                "g4": 0.958,
            }
        }
    }


class GridStabilityOutput(BaseModel):
    """Response payload for grid stability prediction."""
    prediction: int = Field(..., description="Predicted class index: 0 for Stable, 1 for Unstable")
    label: str = Field(..., description="Descriptive stability label: 'Stable' or 'Unstable'")
    probability_stable: float = Field(..., description="Model-estimated probability of the grid being Stable")
    probability_unstable: float = Field(..., description="Model-estimated probability of the grid being Unstable")
    model: Optional[str] = Field(None, description="Identifier of the serving model")


class HealthResponse(BaseModel):
    """Service health status."""
    status: str
    model_loaded: bool
    model_path: str
    version: str


@app.get("/", summary="Root metadata", tags=["Metadata"])
def get_root():
    """Service root describing status and available endpoints."""
    return {
        "service": "Electrical Grid Stability Classification API",
        "status": "operational",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, summary="Service Health Check", tags=["Health"])
def health_check():
    """Verify service liveness and check model pipeline availability."""
    loaded = False
    try:
        _ = load_model()
        loaded = True
    except Exception as e:
        logger.warning(f"Health check model load warning: {e}")

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_path=str(BEST_MODEL_PATH),
        version="1.0.0",
    )


@app.get("/model-info", summary="Model Metadata & Performance", tags=["Model"])
def get_model_info() -> Dict[str, Any]:
    """Retrieve full model metadata, training configuration, hyperparameters, and holdout metrics."""
    metadata = load_metadata()
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata not found. Has `train.py` been executed?",
        )
    return metadata


@app.post(
    "/predict",
    response_model=GridStabilityOutput,
    status_code=status.HTTP_200_OK,
    summary="Predict Grid Stability",
    tags=["Inference"],
)
def predict(payload: GridStabilityInput):
    """Perform real-time binary classification of electrical grid stability."""
    try:
        input_dict = payload.model_dump()
        result = predict_stability(input_dict)
        return GridStabilityOutput(
            prediction=result["prediction"],
            label=result["label"],
            probability_stable=result["probability_stable"],
            probability_unstable=result["probability_unstable"],
            model=result.get("model"),
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact not loaded: {str(e)}",
        )
    except PredictionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input validation error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(e)}",
        )
