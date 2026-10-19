import { useEffect, useMemo, useState } from "react";
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
  X,
} from "lucide-react";
import { average, horizonValues } from "../data";

const HORIZONS = [15, 30, 45, 60];

const TEST_METRICS = [
  { horizon: 15, mae: 3.7314, rmse: 6.5828 },
  { horizon: 30, mae: 4.2100, rmse: 7.5942 },
  { horizon: 45, mae: 4.6255, rmse: 8.4162 },
  { horizon: 60, mae: 4.9884, rmse: 9.0749 },
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
  const [architectureFocus, setArchitectureFocus] = useState("transformer");
  const [analysisTab, setAnalysisTab] = useState("forecast");
  const [architectureDrawerOpen, setArchitectureDrawerOpen] = useState(false);

  useEffect(() => {
    if (!architectureDrawerOpen) return undefined;
    const handleEscape = (event) => {
      if (event.key === "Escape") setArchitectureDrawerOpen(false);
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [architectureDrawerOpen]);

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
            <div role="button" tabIndex={0} aria-pressed={architectureFocus === "graph"} className={`mi-fusion-node mi-fusion-node--interactive ${architectureFocus === "graph" ? "is-active" : ""}`} onClick={() => { setArchitectureFocus("graph"); setAnalysisTab("weather"); setArchitectureDrawerOpen(true); }} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setArchitectureFocus("graph"); setAnalysisTab("weather"); setArchitectureDrawerOpen(true); } }}>
              <div className="mi-fusion-node__icon">
                <GitBranch size={20} />
              </div>

              <div>
                <span>SPATIAL CONTEXT</span>
                <strong>Graph Neural Network</strong>
                <small>207 sensor nodes</small>
              </div>
            </div>

            <div role="button" tabIndex={0} aria-pressed={architectureFocus === "transformer"} className={`mi-fusion-node mi-fusion-node--interactive ${architectureFocus === "transformer" ? "is-active" : ""}`} onClick={() => { setArchitectureFocus("transformer"); setAnalysisTab("weather"); setArchitectureDrawerOpen(true); }} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setArchitectureFocus("transformer"); setAnalysisTab("weather"); setArchitectureDrawerOpen(true); } }}>
              <div className="mi-fusion-node__icon">
                <Layers3 size={20} />
              </div>

              <div>
                <span>TEMPORAL CONTEXT</span>
                <strong>Transformer</strong>
                <small>12 historical timesteps</small>
              </div>
            </div>

            <div role="button" tabIndex={0} aria-pressed={architectureFocus === "weather"} className={`mi-fusion-node mi-fusion-node--interactive ${architectureFocus === "weather" ? "is-active" : ""}`} onClick={() => { setArchitectureFocus("weather"); setAnalysisTab("weather"); setArchitectureDrawerOpen(true); }} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setArchitectureFocus("weather"); setAnalysisTab("weather"); setArchitectureDrawerOpen(true); } }}>
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

      <nav className="mi-analysis-tabs" aria-label="Model analysis">
        {[
          { id: "forecast", label: "Forecast", hint: "Predicted speeds" },
          { id: "accuracy", label: "Accuracy", hint: "Held-out test metrics" },
          { id: "weather", label: "Weather & architecture", hint: "Inputs and components" },
        ].map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={analysisTab === tab.id ? "is-active" : ""}
            aria-pressed={analysisTab === tab.id}
            onClick={() => setAnalysisTab(tab.id)}
          >
            <strong>{tab.label}</strong>
            <small>{tab.hint}</small>
          </button>
        ))}
      </nav>


      <section className={`mi-performance mi-analysis-panel ${analysisTab === "accuracy" ? "is-visible" : "is-hidden"}`} aria-hidden={analysisTab !== "accuracy"}>
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

      {analysisTab === "accuracy" && (
      <details className="mi-compact-panel mi-compact-panel--benchmark">
        <summary className="mi-compact-panel__summary">
          <span><small>MODEL EVALUATION</small><strong>Compare against persistence</strong></span>
          <span className="mi-compact-panel__hint">Expand comparison</span>
        </summary>
      <section className="mi-benchmark">
        <div className="mi-section-heading">
          <div>
            <span className="mi-section-label">BASELINE COMPARISON</span>
            <h2>Does the model outperform persistence?</h2>
          </div>
          <span className="mi-test-badge">
            <ShieldCheck size={15} />
            30-minute held-out test
          </span>
        </div>

        <p className="mi-benchmark-intro">
          Persistence predicts that each sensor will retain its last observed
          speed. The blend combines the model and persistence predictions
          equally, using a weight selected on validation data.
        </p>

        <div className="mi-benchmark-table">
          <div className="mi-benchmark-row mi-benchmark-row--head">
            <span>Method</span>
            <span>MAE (mph)</span>
            <span>RMSE (mph)</span>
          </div>
          <div className="mi-benchmark-row">
            <span>Original GNN–Transformer</span>
            <strong>4.2100</strong>
            <strong>7.5942</strong>
          </div>
          <div className="mi-benchmark-row">
            <span>Persistence baseline</span>
            <strong>4.1759</strong>
            <strong>7.9834</strong>
          </div>
          <div className="mi-benchmark-row mi-benchmark-row--highlight">
            <span>50/50 model–persistence blend</span>
            <strong>3.9189</strong>
            <strong>7.3040</strong>
          </div>
        </div>

        <p className="mi-benchmark-footnote">
          Evaluation uses finite, non-zero observed METR-LA test targets;
          interpolated targets are excluded. The blend results apply only
          to the 30-minute horizon and do not replace the original-model
          metrics above.
        </p>
      </section>
      </details>
      )}



      <section className={`mi-output-grid mi-analysis-panel ${analysisTab === "forecast" ? "is-visible" : "is-hidden"}`} aria-hidden={analysisTab !== "forecast"}>
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
        <article className={`mi-context-card mi-weather-context mi-analysis-panel ${analysisTab === "weather" ? "is-visible" : "is-hidden"}`} aria-hidden={analysisTab !== "weather"}>
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

  
      {architectureDrawerOpen && (
      <div
        className="mi-architecture-drawer-backdrop"
        onMouseDown={(event) => {
          if (event.target === event.currentTarget) setArchitectureDrawerOpen(false);
        }}
      >
      <section className="mi-architecture-inspector mi-architecture-drawer"
        role="dialog" aria-modal="true" aria-label="Explore model architecture">
        <div className="mi-architecture-inspector__header">
          <div>
            <span className="mi-section-label">EXPLORE THE ARCHITECTURE</span>
            <p>Select a component to understand its role in the forecasting pipeline.</p>
          </div>
          <button type="button" className="mi-architecture-drawer__close"
            aria-label="Close architecture details"
            onClick={() => setArchitectureDrawerOpen(false)}>
            <X size={19} />
          </button>
        </div>

        <div className="mi-architecture-inspector__tabs" role="group" aria-label="Model components">
          <button type="button" className={architectureFocus === "graph" ? "is-active" : ""} aria-pressed={architectureFocus === "graph"} onClick={() => setArchitectureFocus("graph")}>Spatial graph</button>
          <button type="button" className={architectureFocus === "transformer" ? "is-active" : ""} aria-pressed={architectureFocus === "transformer"} onClick={() => setArchitectureFocus("transformer")}>Transformer</button>
          <button type="button" className={architectureFocus === "weather" ? "is-active" : ""} aria-pressed={architectureFocus === "weather"} onClick={() => setArchitectureFocus("weather")}>Weather fusion</button>
        </div>

        <div className="mi-architecture-inspector__detail" aria-live="polite">
          {architectureFocus === "graph" && (
            <>
              <span className="mi-section-label">SPATIAL CONTEXT</span>
              <h3>Graph Neural Network</h3>
              <p>Uses the sensor graph to represent spatial relationships that can contribute context to traffic forecasting.</p>
              <div className="mi-architecture-inspector__facts">
                <div><span>Sensor nodes</span><strong>207</strong></div>
                <div><span>Model graph links</span><strong>1,722</strong></div>
                <div><span>Role</span><strong>Spatial context</strong></div>
              </div>
            </>
          )}
          {architectureFocus === "transformer" && (
            <>
              <span className="mi-section-label">TEMPORAL CONTEXT</span>
              <h3>Transformer encoder</h3>
              <p>Processes the historical traffic sequence to learn temporal patterns before producing the forecast horizons.</p>
              <div className="mi-architecture-inspector__facts">
                <div><span>Input window</span><strong>12 timesteps</strong></div>
                <div><span>Attention heads</span><strong>4</strong></div>
                <div><span>Encoder layers</span><strong>2</strong></div>
              </div>
            </>
          )}
          {architectureFocus === "weather" && (
            <>
              <span className="mi-section-label">ENVIRONMENTAL CONTEXT</span>
              <h3>Weather feature fusion</h3>
              <p>Provides environmental context through six weather features, alongside traffic and spatial-temporal representations.</p>
              <div className="mi-architecture-inspector__facts">
                <div><span>Weather features</span><strong>6</strong></div>
                <div><span>Examples</span><strong>Wind, humidity</strong></div>
                <div><span>Role</span><strong>Feature fusion</strong></div>
              </div>
            </>
          )}
        </div>
      </section>
      </div>
      )}

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
