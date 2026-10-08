import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Gauge,
  MapPinned,
  Minus,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import { HORIZONS, trafficState } from "../data";

function behaviorFor(sensor) {
  const first = Number(sensor?.speed15 ?? 0);
  const last = Number(sensor?.speed60 ?? first);
  const delta = last - first;

  const currentState = trafficState(last);

  if (delta < -0.4) {
    if (currentState.key === "pressure") {
      return {
        label: "Pressure deepening",
        tone: "red",
        direction: "down",
      };
    }

    if (currentState.key === "moderate") {
      return {
        label: "Moderate conditions worsening",
        tone: "amber",
        direction: "down",
      };
    }

    return {
      label: "Traffic worsening",
      tone: "amber",
      direction: "down",
    };
  }

  if (delta > 0.4) {
    if (currentState.key === "pressure") {
      return {
        label: "Pressure easing",
        tone: "green",
        direction: "up",
      };
    }

    if (currentState.key === "moderate") {
      return {
        label: "Moderate conditions easing",
        tone: "green",
        direction: "up",
      };
    }

    return {
      label: "Traffic easing",
      tone: "green",
      direction: "up",
    };
  }

  if (currentState.key === "pressure") {
    return {
      label: "Persistent pressure",
      tone: "red",
      direction: "flat",
    };
  }

  if (currentState.key === "moderate") {
    return {
      label: "Persistent moderate conditions",
      tone: "amber",
      direction: "flat",
    };
  }

  return {
    label: "Persistent conditions",
    tone: "amber",
    direction: "flat",
  };
}

function formatDelta(value) {
  if (Math.abs(value) < 0.05) return "0.0";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

function MiniTrajectory({ sensor }) {
  const values = HORIZONS.map((horizon) =>
    Number(sensor?.[`speed${horizon}`] ?? 0)
  );

  // Keep each sensor's real trajectory visible while preventing
  // tiny forecast changes from becoming visually exaggerated.
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1.5);

  return (
    <div className="hotspot-trajectory" aria-label="Forecast trajectory">
      {values.map((value, index) => {
        const state = trafficState(value);
        const normalized = Math.max(0, (value - min) / range);
        const visualProgress = Math.sqrt(normalized);
        const height = 24 + visualProgress * 38;

        return (
          <div
            className="hotspot-trajectory__point"
            key={HORIZONS[index]}
          >
            <span
              className={`hotspot-trajectory__bar is-${state.tone}`}
              style={{ height: `${height}px` }}
              title={`+${HORIZONS[index]} min: ${value.toFixed(1)} mph`}
            />
            <small>+{HORIZONS[index]}</small>
          </div>
        );
      })}
    </div>
  );
}

export default function Hotspots({ sensors, loading }) {
  const [horizon, setHorizon] = useState(15);

  const analysis = useMemo(() => {
    const records = sensors.map((sensor) => ({
      ...sensor,
      speed: Number(sensor?.[`speed${horizon}`] ?? 0),
    }));

    const pressure = records.filter((sensor) => sensor.speed < 45);
    const moving = records.filter(
      (sensor) => sensor.speed >= 45 && sensor.speed < 55
    );
    const freeFlow = records.filter((sensor) => sensor.speed >= 55);
    const severe = records.filter((sensor) => sensor.speed < 30);

    const priority = [...records]
      .sort((a, b) => a.speed - b.speed)
      .slice(0, 6);

    return {
      pressure,
      moving,
      freeFlow,
      severe,
      priority,
      average: records.length
        ? records.reduce((sum, sensor) => sum + sensor.speed, 0) /
          records.length
        : 0,
    };
  }, [sensors, horizon]);

  if (loading) {
    return (
      <section className="page hotspots-page">
        <div className="hotspots-hero">
          <div>
            <span className="hotspots-kicker">
              HOTSPOT INTELLIGENCE
            </span>
            <h1>Find pressure before it becomes a problem.</h1>
            <p>Loading the forecast pressure field…</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="page hotspots-page">
      <header className="hotspots-hero">
        <div className="hotspots-hero__copy">
          <span className="hotspots-kicker">
            <Gauge size={14} />
            Hotspot intelligence
          </span>

          <h1>Find pressure before it becomes a problem.</h1>

          <p>
            UrbanFlow ranks locations with the lowest predicted speeds and
            shows whether that pressure persists, eases, or deepens across
            the forecast horizon.
          </p>
        </div>

        <div className="hotspots-hero__status">
          {analysis.severe.length > 0 ? (
            <AlertTriangle size={18} />
          ) : (
            <ShieldCheck size={18} />
          )}

          <strong>
            {analysis.severe.length > 0
              ? `${analysis.severe.length} severe`
              : "No severe pressure"}
          </strong>

          <span>at +{horizon} min</span>
        </div>
      </header>

      <div className="hotspots-summary">
        <div className="hotspots-summary__item is-below45">
          <span>LOCATIONS BELOW 45 MPH</span>
          <strong>{analysis.pressure.length}</strong>
          <small>below 45 mph</small>
        </div>

        <div className="hotspots-summary__item is-moving">
          <span>MOVING</span>
          <strong>{analysis.moving.length}</strong>
          <small>45–55 mph</small>
        </div>

        <div className="hotspots-summary__item is-free">
          <span>FREE FLOW</span>
          <strong>{analysis.freeFlow.length}</strong>
          <small>55 mph or above</small>
        </div>
      </div>

      <div className="hotspots-toolbar">
        <div>
          <span>FORECAST HORIZON</span>
          <b>Prioritize locations at +{horizon} minutes</b>
        </div>

        <div className="hotspots-tabs">
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

      <div className="hotspots-content">
        <section className="hotspots-queue">
          <div className="hotspots-section-head">
            <div>
              <span>PRIORITY QUEUE</span>
              <h2>Where to look first</h2>
            </div>

            <small>
              {analysis.priority.length} lowest-speed locations
            </small>
          </div>

          <div className="hotspots-list">
            {analysis.priority.map((sensor, index) => {
              const state = trafficState(sensor.speed);
              const behavior = behaviorFor(sensor);

              const delta =
                Number(sensor.speed60 ?? sensor.speed) -
                Number(sensor.speed15 ?? sensor.speed);

              return (
                <article
                  className="hotspot-priority"
                  key={sensor.id}
                >
                  <div className="hotspot-priority__rank">
                    {String(index + 1).padStart(2, "0")}
                  </div>

                  <div
                    className={`hotspot-priority__state is-${state.tone}`}
                  >
                    <span />
                  </div>

                  <div className="hotspot-priority__identity">
                    <div>
                      <strong>{sensor.id}</strong>

                      <span
                        className={`hotspot-behavior is-${behavior.tone}`}
                      >
                        {behavior.direction === "down" && (
                          <TrendingDown size={12} />
                        )}

                        {behavior.direction === "up" && (
                          <TrendingUp size={12} />
                        )}

                        {behavior.direction === "flat" && (
                          <Minus size={12} />
                        )}

                        {behavior.label}
                      </span>
                    </div>

                    <small>
                      {state.label} · {sensor.speed.toFixed(1)} mph at +
                      {horizon} min
                    </small>
                  </div>

                  <MiniTrajectory sensor={sensor} />

                  <div className="hotspot-priority__delta">
                    {delta < -0.4 ? (
                      <ArrowDownRight size={14} />
                    ) : delta > 0.4 ? (
                      <ArrowUpRight size={14} />
                    ) : (
                      <Minus size={14} />
                    )}

                    <strong>{formatDelta(delta)}</strong>
                    <small>mph across horizon</small>
                  </div>


                </article>
              );
            })}
          </div>
        </section>

        <aside className="hotspots-explainer">
          <div className="hotspots-explainer__icon">
            <MapPinned size={17} />
          </div>

          <span>HOW TO READ THIS</span>

          <h2>Hotspots are forecast signals, not incidents.</h2>

          <p>
            A hotspot means UrbanFlow expects lower speed at that location.
            It does not claim that an accident, closure, or other real-world
            event is occurring.
          </p>

          <div className="hotspots-legend">
            <div>
              <i className="is-red" />
              <span>
                <b>Pressure</b>
                <small>below 30 mph</small>
              </span>
            </div>

            <div>
              <i className="is-amber" />
              <span>
                <b>Moderate</b>
                <small>30–45 mph</small>
              </span>
            </div>

            <div>
              <i className="is-blue" />
              <span>
                <b>Moving</b>
                <small>45–55 mph</small>
              </span>
            </div>

            <div>
              <i className="is-green" />
              <span>
                <b>Free flow</b>
                <small>55+ mph</small>
              </span>
            </div>
          </div>

          <div className="hotspots-explainer__footer">
            <AlertTriangle size={14} />
            <span>
              Ranking uses model-predicted speed, not reported incidents.
            </span>
          </div>
        </aside>
      </div>
    </section>
  );
}
