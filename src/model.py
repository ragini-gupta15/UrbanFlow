import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, graph):
        x = torch.matmul(graph, x)
        x = self.linear(x)
        return x


class UrbanFlowFinalModel(nn.Module):

    def __init__(
        self,
        n_sensors=207,
        weather_features=6,
        hidden_dim=128,
        transformer_heads=4,
        transformer_layers=2,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim

        self.gnn_input = GraphConv(
            in_features=1,
            out_features=64
        )

        self.gnn_output = GraphConv(
            in_features=64,
            out_features=hidden_dim
        )

        self.gnn_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_features, 32),
            nn.GELU(),
            nn.Linear(32, hidden_dim),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.head_15 = nn.Linear(hidden_dim, 1)
        self.head_30 = nn.Linear(hidden_dim, 1)
        self.head_45 = nn.Linear(hidden_dim, 1)
        self.head_60 = nn.Linear(hidden_dim, 1)

    def forward(self, traffic, weather, graph):

        B, T, S = traffic.shape

        spatial_sequence = []

        for t in range(T):

            x_t = traffic[:, t, :].unsqueeze(-1)

            x_t = self.gnn_input(
                x_t,
                graph
            )

            x_t = F.gelu(x_t)

            x_t = self.gnn_output(
                x_t,
                graph
            )

            x_t = self.gnn_norm(x_t)
            x_t = F.gelu(x_t)

            spatial_sequence.append(x_t)

        spatial_sequence = torch.stack(
            spatial_sequence,
            dim=1
        )

        temporal_input = (
            spatial_sequence
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B * S, T, self.hidden_dim)
        )

        temporal_output = self.transformer(
            temporal_input
        )

        temporal_features = temporal_output[:, -1, :]

        temporal_features = (
            temporal_features
            .view(B, S, self.hidden_dim)
        )

        weather_features = self.weather_encoder(
            weather[:, -1, :]
        )

        weather_features = weather_features.unsqueeze(1)

        weather_features = weather_features.expand(
            -1,
            S,
            -1
        )

        fused = torch.cat(
            [
                temporal_features,
                weather_features
            ],
            dim=-1
        )

        fused = self.fusion(fused)

        predictions = {
            15: self.head_15(fused).squeeze(-1),
            30: self.head_30(fused).squeeze(-1),
            45: self.head_45(fused).squeeze(-1),
            60: self.head_60(fused).squeeze(-1),
        }

        return predictions