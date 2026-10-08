import os
import joblib

import numpy as np
import torch

from model import UrbanFlowFinalModel


class UrbanFlowPredictor:

    def __init__(self):

        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.model = UrbanFlowFinalModel()

        checkpoint_path = os.path.join(
            self.project_root,
            "models",
            "urbanflow_final_gnn_transformer_weather_multihorizon.pth"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        adjacency_path = os.path.join(
            self.project_root,
            "data",
            "processed",
            "adjacency_matrix.npy"
        )

        adjacency = np.load(adjacency_path)

        adjacency = adjacency.astype(np.float32)

        adjacency = adjacency + np.eye(
            adjacency.shape[0],
            dtype=np.float32
        )

        degree = adjacency.sum(axis=1)

        degree_inv_sqrt = np.zeros_like(
            degree,
            dtype=np.float32
        )

        np.power(
            degree,
            -0.5,
            out=degree_inv_sqrt,
            where=degree > 0
        )

        degree_inv_sqrt[degree == 0] = 0

        adjacency_normalized = (
            degree_inv_sqrt[:, None]
            * adjacency
            * degree_inv_sqrt[None, :]
        )

        self.graph = torch.tensor(
            adjacency_normalized,
            dtype=torch.float32,
            device=self.device
        )

        scaler_path = os.path.join(
            self.project_root,
            "data",
            "processed",
            "traffic_scaler.pkl"
        )

        self.traffic_scaler = joblib.load(
            scaler_path
        )

        self.horizons = [15, 30, 45, 60]

    def predict(self, traffic, weather):

        traffic = np.asarray(
            traffic,
            dtype=np.float32
        )

        weather = np.asarray(
            weather,
            dtype=np.float32
        )

        if traffic.shape != (12, 207):
            raise ValueError(
                "traffic must have shape (12, 207)"
            )

        if weather.shape != (12, 6):
            raise ValueError(
                "weather must have shape (12, 6)"
            )

        traffic_tensor = torch.tensor(
            traffic,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        weather_tensor = torch.tensor(
            weather,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            predictions = self.model(
                traffic_tensor,
                weather_tensor,
                self.graph
            )

        results = {}

        for horizon in self.horizons:

            prediction = (
                predictions[horizon]
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            prediction = self.traffic_scaler.inverse_transform(
                prediction.reshape(1, -1)
            )[0]

            results[horizon] = prediction

        return results