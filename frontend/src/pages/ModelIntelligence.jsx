import { useMemo, useState } from "react";
import {
  BrainCircuit,
  CheckCircle2,
  CloudSun,
  GitBranch,
  Gauge,
  Layers3,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { average, horizonValues } from "../data";

const HORIZONS = [15, 30, 45, 60];

const TEST_METRICS = [
  { horizon: 15, mae: 3.4828, rmse: 6.3331 },
  { horizon: 30, mae: 3.9262, rmse: 7.2986 },
  { horizon: 45, mae: 4.3006, rmse: 8.0625 },
  { horizon: 60, mae: 4.6240, rmse: 8.6722 },
];

function metricFor(horizon) {
  return TEST_METRICS.find((item) => item.horizon === horizon) || TEST_METRICS[0];
}

function directionLabel(change) {
  if (change > 1) return "network slowing";
  if (change < -1) return "network gaining speed";
  return "network holding steady";
}

function directionClass(change) {
  if (change > 1) return "is-down";
  if (change < -1) return "is-up";
  return "is-neutral";
}

export default function ModelIntelligence({ predictions, weather, loading }) {
  const [horizon, setHorizon] = useState(30);

  const forecast = useMemo(
    () =>
      HORIZONS.map((item) => ({
        horizon: item,
        value: average(horizonValues(predictions, item)),
      })),
    [predictions]
  );

  const selectedForecast =
    forecast.find((item) => item.horizon === horizon)?.value || 0;

  const firstForecast = forecast[0]?.value || 0;
  const lastForecast = forecast[forecast.length - 1]?.value || 0;

  const change =
    firstForecast !== 0
      ? ((lastForecast - firstForecast) / firstForecast) * 100
      : 0;

  const selectedMetric = metricFor(horizon);
  const selectedValues = horizonValues(predictions, horizon);

  const predictedMin = selectedValues.length
    ? Math.min(...selectedValues)
    : 0;

  const predictedMax = selectedValues.length
    ? Math.max(...selectedValues)
    : 0;

  const pressureCount = selectedValues.filter((value) => value < 45).length;

  if (loading) {
    return (
      <section className="page model-intelligence-page">
        <div className="model-intelligence-loading">
          Loading model intelligence…
        </div>
      </section>
    );
  }

  return (
    <section className="page model-intelligence-page">

      <header className="mi-hero">
        <div className="mi-hero__copy">
          <span className="mi-eyebrow">MODEL INTELLIGENCE</span>

          <h1>Understand the forecast, not just the number.</h1>

          <p>
            UrbanFlow combines spatial relationships between sensors,
            historical traffic patterns and weather context before producing
            a forecast across four future horizons.
          </p>
        </div>

        <div className={`mi-signal ${directionClass(change)}`}>
          <div className="mi-signal__icon">
            {change > 1 ? (
              <TrendingDown size={19} />
            ) : change < -1 ? (
              <TrendingUp size={19} />
            ) : (
              <Gauge size={19} />
            )}
          </div>

          <div>
            <span>Network signal</span>
            <strong>{directionLabel(change)}</strong>
            <small>{Math.abs(change).toFixed(1)}% from +15 to +60 min</small>
          </div>
        </div>
      </header>

      <div className="mi-horizon-bar">
        <div>
          <span className="mi-section-label">FORECAST HORIZON</span>
          <strong>Inspect the model output</strong>
        </div>

        <div className="mi-horizons">
          {HORIZONS.map((item) => (
            <button
              key={item}
              className={horizon === item ? "is-active" : ""}
              onClick={() => setHorizon(item)}
            >
              +{item}m
            </button>
          ))}
        </div>
      </div>

      <section className="mi-architecture">
        <div className="mi-section-heading">
          <div>
            <span className="mi-section-label">FINAL MODEL</span>
            <h2>Three signals. One forecast.</h2>
          </div>

          <span className="mi-status">
            <CheckCircle2 size={15} />
            Final deployed model
          </span>
        </div>

        <div className="mi-fusion-panel">
          <div className="mi-fusion-inputs">
            <div className="mi-fusion-node">
              <div className="mi-fusion-node__icon">
                <GitBranch size={20} />
              </div>

              <div>
                <span>SPATIAL CONTEXT</span>
                <strong>Graph Neural Network</strong>
                <small>207 sensor nodes</small>
              </div>
            </div>

            <div className="mi-fusion-node">
              <div className="mi-fusion-node__icon">
                <Layers3 size={20} />
              </div>

              <div>
                <span>TEMPORAL CONTEXT</span>
                <strong>Transformer</strong>
                <small>12 historical timesteps</small>
              </div>
            </div>

            <div className="mi-fusion-node">
              <div className="mi-fusion-node__icon">
                <CloudSun size={20} />
              </div>

              <div>
                <span>ENVIRONMENTAL CONTEXT</span>
                <strong>Weather features</strong>
                <small>6 weather features</small>
              </div>
            </div>
          </div>

          <div className="mi-fusion-connectors" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>

          <div className="mi-fusion-core">
            <div className="mi-fusion-core__pulse" />

            <span className="mi-fusion-core__eyebrow">
              URBANFLOW
            </span>

            <h3>Spatial-temporal fusion</h3>

            <p>
              Graph structure, traffic history and environmental context
              are combined before the final multi-horizon prediction.
            </p>

            <div className="mi-fusion-core__tag">
              GNN + Transformer + Weather
            </div>
          </div>

          <div className="mi-fusion-arrow" aria-hidden="true">
            →
          </div>

          <div className="mi-fusion-output">
            <span className="mi-fusion-output__label">
              FORECAST OUTPUT
            </span>

            <div className="mi-fusion-horizons">
              {HORIZONS.map((item) => (
                <button
                  key={item}
                  className={
                    horizon === item
                      ? "mi-fusion-horizon is-active"
                      : "mi-fusion-horizon"
                  }
                  onClick={() => setHorizon(item)}
                >
                  <strong>+{item}m</strong>
                  <small>
                    {forecast
                      .find((entry) => entry.horizon === item)
                      ?.value.toFixed(1)}{" "}
                    mph
                  </small>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mi-fusion-note">
          <BrainCircuit size={16} />
          <span>
            The final model preserves spatial relationships, temporal
            patterns and weather context before producing all four forecast
            horizons.
          </span>
        </div>
      </section>


      <section className="mi-performance">
        <div className="mi-section-heading">
          <div>
            <span className="mi-section-label">MODEL PERFORMANCE</span>
            <h2>How accurately does UrbanFlow forecast?</h2>
          </div>

          <span className="mi-test-badge">
            <ShieldCheck size={15} />
            Held-out test evaluation
          </span>
        </div>

        <div className="mi-performance-layout">
          <div className="mi-metrics-table">
            <div className="mi-table-head">
              <span>Horizon</span>
              <span>MAE</span>
              <span>RMSE</span>
            </div>

            {TEST_METRICS.map((item) => (
              <button
                key={item.horizon}
                className={
                  item.horizon === horizon
                    ? "mi-metric-row is-active"
                    : "mi-metric-row"
                }
                onClick={() => setHorizon(item.horizon)}
              >
                <span>+{item.horizon} min</span>
                <strong>{item.mae.toFixed(4)}</strong>
                <strong>{item.rmse.toFixed(4)}</strong>
              </button>
            ))}
          </div>

          <div className="mi-performance-reading">
            <span className="mi-section-label">SELECTED HORIZON</span>

            <div className="mi-performance-number">
              <strong>{selectedMetric.mae.toFixed(4)}</strong>
              <span>MAE</span>
            </div>

            <div className="mi-performance-secondary">
              <div>
                <span>RMSE</span>
                <strong>{selectedMetric.rmse.toFixed(4)}</strong>
              </div>

              <div>
                <span>Horizon</span>
                <strong>+{horizon} min</strong>
              </div>
            </div>

            <p>
              Error increases as the forecast horizon extends, which is the
              expected trade-off when predicting further into the future.
            </p>
          </div>
        </div>
      </section>

      <section className="mi-output-grid">
        <article className="mi-output-card">
          <div className="mi-section-heading">
            <div>
              <span className="mi-section-label">MODEL OUTPUT</span>
              <h2>Forecast trajectory</h2>
            </div>

            <span className="mi-output-horizon">+{horizon} min</span>
          </div>

          <div className="mi-forecast-chart">
            {forecast.map((item) => {
              const height = Math.max(
                18,
                Math.min(100, (item.value / 70) * 100)
              );

              return (
                <button
                  key={item.horizon}
                  className={
                    item.horizon === horizon
                      ? "mi-forecast-column is-active"
                      : "mi-forecast-column"
                  }
                  onClick={() => setHorizon(item.horizon)}
                >
                  <div className="mi-forecast-value">
                    {item.value.toFixed(1)}
                  </div>

                  <div className="mi-forecast-track">
                    <i style={{ height: `${height}%` }} />
                  </div>

                  <span>+{item.horizon}m</span>
                </button>
              );
            })}
          </div>

          <div className="mi-output-summary">
            <div>
              <span>Selected average</span>
              <strong>{selectedForecast.toFixed(1)} mph</strong>
            </div>

            <div>
              <span>Predicted range</span>
              <strong>
                {predictedMin.toFixed(1)}–{predictedMax.toFixed(1)} mph
              </strong>
            </div>
          </div>
        </article>

        <article className="mi-reading-card">
          <div className="mi-section-heading">
            <div>
              <span className="mi-section-label">MODEL READING</span>
              <h2>What the output tells us</h2>
            </div>

            <BrainCircuit size={19} />
          </div>

          <div className="mi-reading-list">
            <div>
              <span>Locations below 45 mph</span>
              <strong>{pressureCount}</strong>
            </div>

            <div>
              <span>Network direction</span>
              <strong>{directionLabel(change)}</strong>
            </div>

            <div>
              <span>Average at +{horizon}m</span>
              <strong>{selectedForecast.toFixed(1)} mph</strong>
            </div>
          </div>

          <div className="mi-reading-note">
            These statements are derived from model-predicted speeds. They do
            not represent reported road incidents, causes, or guaranteed
            future conditions.
          </div>
        </article>
      </section>

      <section className="mi-transparency-grid">
        <article className="mi-context-card mi-weather-context">
          <div className="mi-context-icon">
            <CloudSun size={20} />
          </div>

          <div className="mi-weather-context__body">
            <span className="mi-section-label">WEATHER CONTEXT</span>
            <h2>Six environmental signals feed the forecast.</h2>
            <p className="mi-weather-context__intro">
              These are the six weather features supplied to UrbanFlow's final
              spatial-temporal model.
            </p>

            <div className="mi-weather-grid">
              <div className="mi-weather-signal">
                <span>Temperature</span>
                <strong>
                  {weather ? `${weather.temperature.toFixed(1)}°` : "—"}
                </strong>
              </div>

              <div className="mi-weather-signal">
                <span>Humidity</span>
                <strong>
                  {weather ? `${weather.humidity.toFixed(0)}%` : "—"}
                </strong>
              </div>

              <div className="mi-weather-signal">
                <span>Precipitation</span>
                <strong>
                  {weather ? weather.precipitation.toFixed(2) : "—"}
                </strong>
              </div>

              <div className="mi-weather-signal">
                <span>Wind speed</span>
                <strong>
                  {weather ? weather.wind_speed.toFixed(1) : "—"}
                </strong>
              </div>

              <div className="mi-weather-signal">
                <span>Pressure</span>
                <strong>
                  {weather ? weather.pressure.toFixed(1) : "—"}
                </strong>
              </div>

              <div className="mi-weather-signal">
                <span>Cloud cover</span>
                <strong>
                  {weather ? weather.cloud_cover.toFixed(1) : "—"}
                </strong>
              </div>
            </div>

            <div className="mi-weather-model-note">
              <span>6 FEATURES</span>
              <span>→</span>
              <b>Spatial-temporal fusion</b>
              <span>→</span>
              <b>Traffic forecast</b>
            </div>
          </div>
        </article>

        <article className="mi-transparency-card">
          <span className="mi-section-label">MODEL TRANSPARENCY</span>
          <h2>
            Interpretation without pretending the model said more than it did.
          </h2>

          <p>
            UrbanFlow's final system combines GNN spatial context, Transformer
            temporal modelling and six weather features to produce 15, 30, 45
            and 60-minute forecasts.
          </p>

          <div className="mi-transparency-points">
            <span>
              <CheckCircle2 size={15} />
              Real model outputs
            </span>

            <span>
              <CheckCircle2 size={15} />
              Real test metrics
            </span>

            <span>
              <CheckCircle2 size={15} />
              No fabricated confidence scores
            </span>
          </div>

          <div className="mi-transparency-detail-grid">
            <div className="mi-transparency-detail">
              <span>MODEL USES</span>
              <strong>Spatial + temporal + environmental context</strong>
              <small>207 sensors · 12 historical timesteps · 6 weather features</small>
            </div>

            <div className="mi-transparency-detail">
              <span>MODEL PRODUCES</span>
              <strong>Four future traffic forecasts</strong>
              <small>+15 · +30 · +45 · +60 minute horizons</small>
            </div>

            <div className="mi-transparency-detail">
              <span>MODEL DOES NOT CLAIM</span>
              <strong>Causes or guaranteed conditions</strong>
              <small>Not incident detection · not causal explanation · not certainty</small>
            </div>
          </div>
        </article>
      </section>
    </section>
  );
}
