from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np



from inference import UrbanFlowPredictor
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="UrbanFlow API",
    description="Traffic forecasting API using GNN, Transformer and weather features.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = UrbanFlowPredictor()


class PredictionRequest(BaseModel):
    traffic: list[list[float]]
    weather: list[list[float]]


@app.get("/")
def root():
    return {
        "project": "UrbanFlow",
        "status": "running",
        "horizons": [15, 30, 45, 60]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "UrbanFlow Final Model",
        "device": str(predictor.device)
    }
@app.get("/demo-input")
def demo_input():
    traffic_data = np.load(
        "data/processed/test_sequences.npz"
    )["X"]

    weather_data = np.load(
        "data/processed/weather_test_sequences.npy"
    )

    return {
        "traffic": traffic_data[0].tolist(),
        "weather": weather_data[0].tolist()
    }

_demo_predictions = None


@app.get("/demo-predict")
def demo_predict():
    global _demo_predictions

    if _demo_predictions is not None:
        return _demo_predictions

    traffic_data = np.load(
        "data/processed/test_sequences.npz"
    )["X"]

    weather_data = np.load(
        "data/processed/weather_test_sequences.npy"
    )

    traffic = traffic_data[0].astype(np.float32)
    weather = weather_data[0].astype(np.float32)

    predictions = predictor.predict(
        traffic,
        weather
    )

    _demo_predictions = {
        "15_min": predictions[15].tolist(),
        "30_min": predictions[30].tolist(),
        "45_min": predictions[45].tolist(),
        "60_min": predictions[60].tolist()
    }

    return _demo_predictions
@app.get("/weather-data")
def weather_data():
    weather_data = np.load(
        "data/processed/weather_test_sequences.npy"
    )

    sample = weather_data[0, -1]

    return {
        "temperature": float(sample[0]),
        "humidity": float(sample[1]),
        "precipitation": float(sample[2]),
        "wind_speed": float(sample[3]),
        "pressure": float(sample[4]),
        "cloud_cover": float(sample[5])
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    traffic = np.asarray(
        request.traffic,
        dtype=np.float32
    )

    weather = np.asarray(
        request.weather,
        dtype=np.float32
    )

    if traffic.shape != (12, 207):
        raise HTTPException(
            status_code=400,
            detail="traffic must have shape (12, 207)"
        )

    if weather.shape != (12, 6):
        raise HTTPException(
            status_code=400,
            detail="weather must have shape (12, 6)"
        )

    predictions = predictor.predict(
        traffic,
        weather
    )

    return {
        "15_min": predictions[15].tolist(),
        "30_min": predictions[30].tolist(),
        "45_min": predictions[45].tolist(),
        "60_min": predictions[60].tolist()
    }