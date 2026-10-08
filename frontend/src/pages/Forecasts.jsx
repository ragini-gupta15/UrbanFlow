import { useMemo, useState } from "react";
import {
  Activity,
  ArrowDown,
  ArrowUp,
  Clock3,
  Gauge,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";
import {
  average,
  HORIZONS,
  horizonValues,
  maxValue,
  minValue,
} from "../data";
import PageHeader from "../components/PageHeader";

export default function Forecasts({ predictions, loading }) {
  const [active, setActive] = useState(30);
  const [selectedSensor, setSelectedSensor] = useState(0);

  const rows = useMemo(
    () =>
      HORIZONS.map((horizon) => {
        const values = horizonValues(predictions, horizon);

        return {
          horizon,
          values,
          average: average(values),
          min: minValue(values),
          max: maxValue(values),
        };
      }),
    [predictions]
  );

  const current =
    rows.find((row) => row.horizon === active) || rows[1];

  const selectedTrajectory = HORIZONS.map((horizon) => ({
    horizon,
    speed: Number(
      horizonValues(predictions, horizon)[selectedSensor] ?? 0
    ),
  }));

  const selectedFirst = selectedTrajectory[0]?.speed || 0;
  const selectedLast =
    selectedTrajectory[selectedTrajectory.length - 1]?.speed || 0;

  const selectedDelta = selectedLast - selectedFirst;

  const selectedState =
    selectedLast < 30
      ? "Pressure"
      : selectedLast < 45
        ? "Moderate"
        : selectedLast < 55
          ? "Moving"
          : "Free flow";

  const baseline = rows[0]?.average || 0;

  const change = baseline
    ? ((current.average - baseline) / baseline) * 100
    : 0;

  const previous =
    rows.find((row) => row.horizon === active - 15)?.average ||
    baseline;

  const delta = current.average - previous;

  const pressure =
    current.values.length > 0
      ? current.values.filter((value) => value < 30).length
      : 0;

  const moderate =
    current.values.length > 0
      ? current.values.filter(
          (value) => value >= 30 && value < 45
        ).length
      : 0;

  const free =
    current.values.length > 0
      ? current.values.filter((value) => value >= 45).length
      : 0;

  const timelineWidth =
    ((active - 15) / 45) * 100;

  const sensorField = current.values.slice(0, 207);

  if (loading) {
    return (
      <section className="page">
        <PageHeader
          eyebrow="Forecast studio"
          title="Loading the future..."
          description="UrbanFlow is preparing the multi-horizon prediction field."
        />
      </section>
    );
  }

  return (
    <section className="page forecast-machine">

      <PageHeader
        eyebrow="Forecast time machine"
        title="See traffic before it happens."
        description="Move through the next hour and watch UrbanFlow's prediction field evolve across all 207 monitored sensors."
        action={
          <div className="forecast-mode">
            <Clock3 size={15} />
            Live prediction engine
          </div>
        }
      />

      <div className="forecast-machine__hero">

        <div className="forecast-machine__topline">
          <div>
            <span className="forecast-kicker">
              PREDICTION HORIZON
            </span>

            <div className="forecast-big-time">
              +{active}
              <small>MIN</small>
            </div>
          </div>

          <div className="forecast-direction">
            {change >= 0 ? (
              <TrendingUp size={18} />
            ) : (
              <TrendingDown size={18} />
            )}

            <div>
              <strong>
                {change >= 0 ? "+" : ""}
                {change.toFixed(1)}%
              </strong>

              <span>
                vs current network
              </span>
            </div>
          </div>
        </div>

        <div className="forecast-timeline">

          <div
            className="forecast-timeline__progress"
            style={{ width: `${timelineWidth}%` }}
          />

          {HORIZONS.map((horizon) => (
            <button
              key={horizon}
              className={
                active === horizon
                  ? "forecast-time-node is-active"
                  : "forecast-time-node"
              }
              onClick={() => setActive(horizon)}
            >
              <span className="forecast-time-node__dot">
                {active === horizon && <i />}
              </span>

              <strong>
                +{horizon}
              </strong>

              <small>
                {horizon === 15
                  ? "near future"
                  : horizon === 60
                    ? "far future"
                    : "forecast"}
              </small>
            </button>
          ))}

        </div>

        <div className="forecast-machine__hint">
          <Zap size={14} />
          Select a horizon to travel through the prediction
        </div>

      </div>

      <div className="forecast-live-grid">

        <div className="forecast-speed-panel">

          <div className="forecast-panel-heading">
            <div>
              <span>NETWORK SPEED FIELD</span>
              <h3>
                How the city is expected to move
              </h3>
            </div>

            <div className="forecast-live-badge">
              <i />
              PREDICTED
            </div>
          </div>

          <div key={active} className={`sensor-field sensor-field--horizon-${active}`}>

            <svg
              className="sensor-field__flow"
              viewBox="0 0 1000 220"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <defs>
                <linearGradient id="forecastFlowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#45e6a8" stopOpacity="0" />
                  <stop offset="22%" stopColor="#45e6a8" stopOpacity=".55" />
                  <stop offset="50%" stopColor="#6fe7ff" stopOpacity=".9" />
                  <stop offset="78%" stopColor="#45e6a8" stopOpacity=".55" />
                  <stop offset="100%" stopColor="#45e6a8" stopOpacity="0" />
                </linearGradient>

                <filter id="forecastFlowGlow">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              <g className="forecast-flow-lanes" filter="url(#forecastFlowGlow)">
                <path d="M-40 48 C150 18 230 72 390 45 S690 20 1040 58" />
                <path d="M-40 92 C120 125 245 60 410 102 S710 132 1040 88" />
                <path d="M-40 138 C145 105 270 168 445 132 S735 105 1040 150" />
                <path d="M-40 178 C150 202 280 150 430 180 S720 200 1040 166" />
              </g>

              <g className="forecast-flow-packets">
                <circle cx="80" cy="42" r="3" />
                <circle cx="350" cy="99" r="3" />
                <circle cx="620" cy="126" r="3" />
                <circle cx="850" cy="170" r="3" />
              </g>
            </svg>

            {sensorField.map((value, index) => {
              const normalized = Math.max(
                0,
                Math.min(1, value / 70)
              );

              const state =
                value < 30
                  ? "pressure"
                  : value < 45
                    ? "moderate"
                    : value < 55
                      ? "moving"
                      : "free";

              return (
                <button
                  key={index}
                  className={`sensor-field__dot is-${state} ${
                    selectedSensor === index
                      ? "is-selected"
                      : ""
                  }`}
                  style={{
                    "--speed": normalized,
                    "--delay": `${(index % 30) * 0.025}s`,
                  }}
                  title={`Sensor ${String(index + 1).padStart(
                    3,
                    "0"
                  )}: ${value.toFixed(1)} mph`}
                  onClick={() => setSelectedSensor(index)}
                  aria-label={`Sensor ${String(index + 1).padStart(
                    3,
                    "0"
                  )}, ${value.toFixed(1)} miles per hour`}
                />
              );
            })}

          </div>

          <div className="sensor-field__legend">
            <span>
              <i className="is-pressure" />
              Pressure
            </span>

            <span>
              <i className="is-moderate" />
              Moderate
            </span>

            <span>
              <i className="is-moving" />
              Moving
            </span>

            <span>
              <i className="is-free" />
              Free flow
            </span>

            <span>
              207 sensors
            </span>
          </div>

        </div>

        <div className="forecast-speed-card">

          <span className="forecast-kicker">
            PREDICTED NETWORK SPEED
          </span>

          <div className="forecast-speed-value">
            {current.average.toFixed(1)}
            <small>mph</small>
          </div>

          <div
            className={
              delta >= 0
                ? "forecast-delta is-positive"
                : "forecast-delta is-negative"
            }
          >
            {delta >= 0 ? (
              <ArrowUp size={15} />
            ) : (
              <ArrowDown size={15} />
            )}

            {delta >= 0 ? "+" : ""}
            {delta.toFixed(1)} mph
            <span>from previous horizon</span>
          </div>

          <div className="forecast-gauge">
            <div
              className="forecast-gauge__fill"
              style={{
                width: `${Math.min(
                  100,
                  (current.average / 70) * 100
                )}%`,
              }}
            />
          </div>

          <div className="forecast-gauge__labels">
            <span>0</span>
            <span>35</span>
            <span>70 mph</span>
          </div>

        </div>

        <div className="forecast-sensor-trajectory">

          <div className="forecast-sensor-trajectory__header">
            <div>
              <span className="forecast-kicker">SELECTED SENSOR</span>
              <strong>
                S-{String(selectedSensor + 1).padStart(3, "0")}
              </strong>
            </div>

            <span
              className={`forecast-sensor-trajectory__state is-${selectedState
                .toLowerCase()
                .replace(" ", "-")}`}
            >
              {selectedState}
            </span>
          </div>

          <div className="forecast-sensor-trajectory__summary">
            <div>
              <strong>{selectedLast.toFixed(1)}</strong>
              <small>mph at +60m</small>
            </div>

            <span
              className={
                selectedDelta <= 0
                  ? "is-down"
                  : "is-up"
              }
            >
              {selectedDelta < 0 ? "↓" : selectedDelta > 0 ? "↑" : "→"}{" "}
              {Math.abs(selectedDelta).toFixed(1)} mph
            </span>
          </div>

          <div className="forecast-sensor-trajectory__chart">

            <div className="forecast-trajectory-grid">
              <span>70</span>
              <span>45</span>
              <span>30</span>
              <span>0</span>
            </div>

            <div className="forecast-trajectory-line">

              {selectedTrajectory.map((point, index) => {

                const speeds = selectedTrajectory.map(
                  (item) => item.speed
                );

                const min = Math.min(...speeds);
                const max = Math.max(...speeds);
                const range = Math.max(max - min, 1);

                const left =
                  `${(index / (selectedTrajectory.length - 1)) * 100}%`;

                const bottom =
                  `${12 + ((point.speed - min) / range) * 64}%`;

                return (
                  <div
                    key={point.horizon}
                    className="forecast-trajectory-point"
                    style={{
                      left,
                      bottom,
                    }}
                    title={`+${point.horizon} min · ${point.speed.toFixed(1)} mph`}
                  >
                    <span />
                    <label>{point.speed.toFixed(1)}</label>
                  </div>
                );
              })}

              <svg
                className="forecast-trajectory-svg"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                <defs>
                  <linearGradient
                    id="forecastTrajectoryGradient"
                    x1="0%"
                    y1="0%"
                    x2="100%"
                    y2="0%"
                  >
                    <stop offset="0%" stopColor="#45e6a8" />
                    <stop offset="55%" stopColor="#6fe7ff" />
                    <stop offset="100%" stopColor="#ffb347" />
                  </linearGradient>

                  <filter id="forecastTrajectoryGlow">
                    <feGaussianBlur stdDeviation="1.8" result="blur" />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                </defs>

                <polyline
                  points={selectedTrajectory
                    .map((point, index) => {
                      const speeds = selectedTrajectory.map(
                        (item) => item.speed
                      );
                      const min = Math.min(...speeds);
                      const max = Math.max(...speeds);
                      const range = Math.max(max - min, 1);

                      const x =
                        (index /
                          (selectedTrajectory.length - 1)) *
                        100;

                      const y =
                        82 -
                        ((point.speed - min) / range) * 64;

                      return `${x},${y}`;
                    })
                    .join(" ")}
                  fill="none"
                  stroke="url(#forecastTrajectoryGradient)"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                  filter="url(#forecastTrajectoryGlow)"
                  className="forecast-trajectory-path"
                />
              </svg>

            </div>

            <div className="forecast-trajectory-horizons">
              {selectedTrajectory.map((point) => (
                <span key={point.horizon}>
                  +{point.horizon}m
                </span>
              ))}
            </div>

          </div>

          <div className="forecast-sensor-trajectory__hint">
            Click a sensor above to inspect its predicted movement.
          </div>

        </div>

      </div>

      <div className="forecast-stats-grid">

        <div className="forecast-stat-card">
          <div className="forecast-stat-icon is-red">
            <Gauge size={17} />
          </div>

          <div>
            <span>CONGESTED</span>
            <strong>{pressure}</strong>
            <small>sensors below 30 mph</small>
          </div>
        </div>

        <div className="forecast-stat-card">
          <div className="forecast-stat-icon is-amber">
            <Activity size={17} />
          </div>

          <div>
            <span>MODERATE</span>
            <strong>{moderate}</strong>
            <small>sensors between 30–45 mph</small>
          </div>
        </div>

        <div className="forecast-stat-card">
          <div className="forecast-stat-icon is-green">
            <Zap size={17} />
          </div>

          <div>
            <span>FREE FLOW</span>
            <strong>{free}</strong>
            <small>sensors 55 mph or above</small>
          </div>
        </div>

        <div className="forecast-stat-card">
          <div className="forecast-stat-icon is-cyan">
            <Clock3 size={17} />
          </div>

          <div>
            <span>FORECAST RANGE</span>
            <strong>
              {current.min.toFixed(0)}–{current.max.toFixed(0)}
            </strong>
            <small>predicted mph range</small>
          </div>
        </div>

      </div>

      <div className="forecast-evolution">

        <div className="forecast-evolution__header">
          <div>
            <span className="forecast-kicker">
              NETWORK EVOLUTION
            </span>

            <h3>
              Where is traffic heading?
            </h3>
          </div>

          <div className="forecast-evolution__answer">
            {change > 1
              ? "Traffic is expected to accelerate"
              : change < -1
                ? "Traffic is expected to slow"
                : "Traffic is expected to remain stable"}
          </div>
        </div>

        <div className="forecast-evolution__rows">

          {rows.map((row, index) => {
            const width = Math.max(
              8,
              Math.min(100, (row.average / 70) * 100)
            );

            return (
              <button
                key={row.horizon}
                className={
                  active === row.horizon
                    ? "evolution-row is-active"
                    : "evolution-row"
                }
                onClick={() => setActive(row.horizon)}
              >

                <span className="evolution-row__time">
                  +{row.horizon}
                  <small>min</small>
                </span>

                <span className="evolution-row__track">
                  <i style={{ width: `${width}%` }} />
                </span>

                <strong>
                  {row.average.toFixed(1)}
                  <small>mph</small>
                </strong>

                <span className="evolution-row__change">
                  {index === 0
                    ? "baseline"
                    : row.average >=
                        rows[index - 1].average
                      ? "+"
                      : ""}
                  {index === 0
                    ? ""
                    : (
                        row.average -
                        rows[index - 1].average
                      ).toFixed(1)}
                </span>

              </button>
            );
          })}

        </div>

      </div>

      <div className="forecast-explanation">
        <div className="forecast-explanation__icon">
          <Activity size={18} />
        </div>

        <div>
          <strong>
            What UrbanFlow is predicting
          </strong>

          <p>
            The final GNN + Transformer model combines
            spatial traffic relationships, temporal patterns
            and weather context to forecast speed at every
            monitored sensor across four future horizons.
          </p>
        </div>

        <div className="forecast-explanation__badge">
          207
          <span>SENSORS</span>
        </div>

      </div>

    </section>
  );
}
