from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np



from src.inference import UrbanFlowPredictor
from fastapi.middleware.cors import CORSMiddleware
import os



app = FastAPI(
    title="UrbanFlow API",
    description="Traffic forecasting API using GNN, Transformer and weather features.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "URBANFLOW_ALLOWED_ORIGINS",
            "http://localhost:5173,http://localhost:5174,http://127.0.0.1:4173",
        ).split(",")
        if origin.strip()
    ],
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
    project_root = predictor.project_root

    traffic_data = np.load(
        os.path.join(project_root, "data", "processed", "test_sequences.npz")
    )["X"]

    weather_data = np.load(
        os.path.join(project_root, "data", "processed", "weather_test_sequences.npy")
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

    project_root = predictor.project_root
    traffic_path = os.path.join(
        project_root, "data", "processed", "test_sequences.npz"
    )
    weather_path = os.path.join(
        project_root, "data", "processed", "weather_test_sequences.npy"
    )

    if os.path.isfile(traffic_path) and os.path.isfile(weather_path):
        traffic_data = np.load(traffic_path)["X"]
        weather_data = np.load(weather_path)

        predictions = predictor.predict(
            traffic_data[0].astype(np.float32),
            weather_data[0].astype(np.float32),
        )

        _demo_predictions = {
            "15_min": predictions[15].tolist(),
            "30_min": predictions[30].tolist(),
            "45_min": predictions[45].tolist(),
            "60_min": predictions[60].tolist(),
        }
        return _demo_predictions

    replay_path = os.path.join(
        project_root, "frontend", "public", "demo-replay-samples.json"
    )

    if not os.path.isfile(replay_path):
        raise HTTPException(
            status_code=503,
            detail="Test datasets and saved demo predictions are unavailable",
        )

    import json

    with open(replay_path, "r", encoding="utf-8") as replay_file:
        replay_data = json.load(replay_file)

    samples = replay_data.get("samples", [])
    if not samples:
        raise HTTPException(
            status_code=500,
            detail="Saved demo predictions are empty",
        )

    _demo_predictions = samples[0]["predictions"]
    return _demo_predictions

@app.get("/weather-data")
def weather_data():
    project_root = predictor.project_root

    weather_data = np.load(
        os.path.join(
            project_root,
            "data",
            "processed",
            "weather_test_sequences.npy",
        )
    )

    sample = weather_data[0, -1]

    return {
        "temperature": float(sample[0]),
        "humidity": float(sample[1]),
        "precipitation": float(sample[2]),
        "wind_speed": float(sample[3]),
        "pressure": float(sample[4]),
        "cloud_cover": float(sample[5]),
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

@app.get("/demo-replay")
def demo_replay(sample_index: int = 0):
    """Use model inference when datasets exist; otherwise serve historical replay."""
    project_root = predictor.project_root

    traffic_path = os.path.join(
        project_root, "data", "processed", "test_sequences.npz"
    )
    weather_path = os.path.join(
        project_root, "data", "processed", "weather_test_sequences.npy"
    )

    if os.path.isfile(traffic_path) and os.path.isfile(weather_path):
        traffic_data = np.load(traffic_path)["X"]
        weather_sequences = np.load(weather_path)
        sample_count = min(len(traffic_data), len(weather_sequences))

        if sample_count < 1:
            raise HTTPException(
                status_code=500, detail="Historical replay dataset is empty"
            )

        index = sample_index % sample_count
        traffic = traffic_data[index].astype(np.float32)
        weather_sequence = weather_sequences[index].astype(np.float32)

        if traffic.shape != (12, 207) or weather_sequence.shape != (12, 6):
            raise HTTPException(
                status_code=500, detail="Unexpected replay sample shape"
            )

        predictions = predictor.predict(traffic, weather_sequence)
        weather_sample = weather_sequence[-1]

        return {
            "sample_index": int(index),
            "sample_count": int(sample_count),
            "predictions": {
                "15_min": predictions[15].tolist(),
                "30_min": predictions[30].tolist(),
                "45_min": predictions[45].tolist(),
                "60_min": predictions[60].tolist(),
            },
            "weather": {
                "temperature": float(weather_sample[0]),
                "humidity": float(weather_sample[1]),
                "precipitation": float(weather_sample[2]),
                "wind_speed": float(weather_sample[3]),
                "pressure": float(weather_sample[4]),
                "cloud_cover": float(weather_sample[5]),
            },
        }

    replay_path = os.path.join(
        project_root, "frontend", "public", "demo-replay-samples.json"
    )
    if not os.path.isfile(replay_path):
        raise HTTPException(
            status_code=503,
            detail="Replay datasets and historical fallback are unavailable",
        )

    import json

    with open(replay_path, "r", encoding="utf-8") as replay_file:
        replay_data = json.load(replay_file)

    samples = replay_data.get("samples", [])
    if not samples:
        raise HTTPException(
            status_code=500, detail="Historical replay samples are empty"
        )

    sample = samples[sample_index % len(samples)]
    return {
        "sample_index": sample["sample_index"],
        "sample_count": sample["sample_count"],
        "predictions": sample["predictions"],
        "weather": sample["weather"],
    }