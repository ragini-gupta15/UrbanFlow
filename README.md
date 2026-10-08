# UrbanFlow

## Spatiotemporal Traffic Forecasting with GNNs, Transformers & Weather

UrbanFlow is an end-to-end traffic forecasting system that predicts traffic speed across a network of 207 road sensors using historical traffic patterns, learned spatial relationships, and weather context.

## Final System

- Graph Neural Network (GNN) for spatial relationships
- Transformer for temporal dependencies
- 6 weather features
- Multi-horizon forecasting
- FastAPI inference backend
- React + MapLibre interactive dashboard

## Final Model

```text
Historical Traffic
12 Timesteps × 207 Sensors
        │
        ├── Graph Structure ──► GNN
        │
        └── Temporal Sequence ► Transformer
                    │
              Spatial-Temporal
                  Fusion
                    │
              + Weather Context
                    │
             Multi-Horizon Head
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
     +15          +30          +45          +60 min

## Final Test Performance

| Horizon | MAE | RMSE |
|---|---:|---:|
| 15 min | 3.4828 mph | 6.3331 mph |
| 30 min | 3.9262 mph | 7.2986 mph |
| 45 min | 4.3006 mph | 8.0625 mph |
| 60 min | 4.6240 mph | 8.6722 mph |

These metrics come from the final untouched test evaluation.

## Traffic Network

UrbanFlow models:

- 207 traffic sensors
- 1,722 directed graph connections
- Real sensor coordinates
- Historical traffic-speed observations

## Model Development

The project was developed through:

1. Data understanding
2. Data preprocessing
3. Baseline models
4. LSTM modelling
5. Graph construction
6. GNN modelling
7. Transformer modelling
8. UrbanFlow architecture experiments
9. Congestion classification
10. Transformer and spatial-temporal optimization
11. Final GNN + Transformer + Weather multi-horizon model

Experiments include Transformer optimization, spatial-temporal fusion, graph-aware modelling, Huber loss, weather integration, weather ablation, and multi-horizon forecasting.

Experimental results are preserved under `results/experiments/`.

## Dashboard

UrbanFlow includes five interactive views:

- **Command Center** — operational traffic overview
- **Forecasts** — future traffic predictions
- **Network** — sensor and graph relationships
- **Hotspots** — locations requiring attention
- **Model Intelligence** — model architecture, performance and limitations

## Technology Stack

### Machine Learning

Python, PyTorch, NumPy, Pandas, Scikit-learn, Jupyter

### Backend

FastAPI, Uvicorn, PyTorch

### Frontend

React, Vite, MapLibre GL, Recharts, Framer Motion, Lucide React, Tailwind CSS

## Project Structure

```text
Urbanflow/
├── data/
├── models/
│   └── urbanflow_final_gnn_transformer_weather_multihorizon.pth
├── notebooks/
├── results/
├── src/
│   ├── api.py
│   ├── inference.py
│   └── model.py
├── frontend/
├── .gitignore
└── README.md
