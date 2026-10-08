# UrbanFlow

UrbanFlow is a traffic forecasting system built to predict vehicle speeds across a network of 207 traffic sensors.

The project combines historical traffic data, the spatial relationships between sensors, and weather information to produce forecasts for the next 15, 30, 45 and 60 minutes.

The repository contains the model development notebooks, trained model, evaluation results, FastAPI inference service and the React dashboard.

## Final Model

The final UrbanFlow model combines:

- Graph Neural Network for spatial relationships between sensors
- Transformer for temporal traffic patterns
- Weather features for additional context
- Multi-horizon prediction for four future time intervals

Input configuration:

- Sensors: 207
- Historical timesteps: 12
- Historical window: 60 minutes
- Weather features: 6
- Forecast horizons: 15, 30, 45 and 60 minutes

The traffic graph contains 1,722 directed connections between the 207 sensors.

## Test Results

The final model was evaluated on an untouched test set.

| Horizon | MAE | RMSE |
| --- | ---: | ---: |
| 15 min | 3.4828 mph | 6.3331 mph |
| 30 min | 3.9262 mph | 7.2986 mph |
| 45 min | 4.3006 mph | 8.0625 mph |
| 60 min | 4.6240 mph | 8.6722 mph |

These results are taken from the final multi-horizon test evaluation.

## Model Development

UrbanFlow was developed progressively rather than starting directly with the final architecture.

The notebooks cover:

1. Data understanding
2. Data preprocessing
3. Baseline models
4. LSTM modelling
5. Graph construction
6. GNN modelling
7. Transformer modelling
8. Spatial-temporal experiments
9. Congestion classification
10. Weather integration
11. Final GNN + Transformer multi-horizon model

The baseline and experimental models are retained in the repository results so that the development process can be inspected rather than only showing the final model.

## Traffic Network

The model works with an actual sensor network rather than treating every sensor as an independent time series.

The graph was constructed from the sensor locations and contains:

- 207 sensors
- 1,722 directed model connections
- Real sensor coordinates
- Historical traffic-speed observations

The same sensor locations and network structure are also used by the dashboard.

## Dashboard

UrbanFlow includes an interactive dashboard for exploring the model and the traffic network.

The dashboard is divided into five views:

### Command Center

A high-level view of the current traffic situation and key network metrics.

### Forecasts

Shows predicted traffic conditions across the available forecast horizons.

### Network

Displays the sensor network and the relationships between locations.

### Hotspots

Highlights locations where traffic conditions require closer attention.

### Model Intelligence

Provides information about the model architecture, performance and limitations.

The frontend uses the real sensor coordinates and a visual representation of the traffic graph.

## Technology

Python, PyTorch, NumPy, Pandas, Scikit-learn, Jupyter, FastAPI, Uvicorn, React, Vite, MapLibre GL, Recharts, Framer Motion, Lucide React and Tailwind CSS.


## Project Structure

```text
UrbanFlow/
├── data/
├── models/
│   └── urbanflow_final_gnn_transformer_weather_multihorizon.pth
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── ...
│   └── 11_urbanflow_final_model.ipynb
├── results/
│   └── experiments/
├── src/
│   ├── api.py
│   ├── inference.py
│   └── model.py
├── frontend/
├── requirements.txt
├── requirements-notebooks.txt
├── .gitignore
└── README.md