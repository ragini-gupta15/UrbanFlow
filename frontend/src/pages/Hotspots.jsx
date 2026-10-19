import { useEffect, useMemo, useState } from "react";
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
  Search,
  X,
  ArrowUpDown,
  ChevronRight,
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

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1.5);

  const points = values.map((value, index) => ({
    x: 12 + index * 52,
    y: 40 - ((value - min) / range) * 27,
    value,
    horizon: HORIZONS[index],
    state: trafficState(value),
  }));

  const polyline = points.map((point) => `${point.x},${point.y}`).join(" ");

  return (
    <div className="hotspot-trajectory-chart">
      <svg
        viewBox="0 0 180 56"
        role="img"
        aria-label={`Predicted speeds: ${points
          .map((point) => `plus ${point.horizon} minutes, ${point.value.toFixed(1)} miles per hour`)
          .join("; ")}`}
      >
        {[12, 38, 64].map((y) => (
          <line
            key={y}
            x1="5"
            x2="175"
            y1={y}
            y2={y}
            className="hotspot-trajectory-chart__grid"
          />
        ))}
        <polyline
          points={polyline}
          className="hotspot-trajectory-chart__line"
        />
        {points.map((point) => (
          <circle
            key={point.horizon}
            cx={point.x}
            cy={point.y}
            r="3.4"
            className={`hotspot-trajectory-chart__dot is-${point.state.tone}`}
          >
            <title>
              +{point.horizon} min: {point.value.toFixed(1)} mph
            </title>
          </circle>
        ))}
      </svg>
      <div className="hotspot-trajectory-chart__labels">
        {points.map((point) => (
          <span key={point.horizon}>
            <b>{point.value.toFixed(0)}</b>
            <small>+{point.horizon}m</small>
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Hotspots({ sensors = [], loading }) {
  const [horizon, setHorizon] = useState(15);
  const [filter, setFilter] = useState("all");
  const [sort, setSort] = useState("speed");
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [showAll, setShowAll] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);

  useEffect(() => {
    if (!inspectorOpen) return undefined;

    const handleKeyDown = (event) => {
      if (event.key === "Escape") setInspectorOpen(false);
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [inspectorOpen]);

  const records = useMemo(() => sensors.map((sensor) => ({
    ...sensor,
    speed: Number(sensor?.[`speed${horizon}`] ?? 0),
    trend: Number(sensor?.speed60 ?? sensor?.speed ?? 0) -
      Number(sensor?.speed15 ?? sensor?.speed ?? 0),
  })), [sensors, horizon]);

  const analysis = useMemo(() => {
    const pressure = records.filter((s) => s.speed < 45);
    const moving = records.filter((s) => s.speed >= 45 && s.speed < 55);
    const freeFlow = records.filter((s) => s.speed >= 55);
    const severe = records.filter((s) => s.speed < 30);

    return {
      pressure, moving, freeFlow, severe,
      average: records.length
        ? records.reduce((sum, s) => sum + s.speed, 0) / records.length
        : 0,
    };
  }, [records]);

  const filtered = useMemo(() => {
    let result = records.filter((sensor) => {
      const state = trafficState(sensor.speed);
      const key = sensor.speed < 30 ? "pressure"
        : sensor.speed < 45 ? "moderate"
        : sensor.speed < 55 ? "moving" : "free";
      const matchesFilter = filter === "all" || filter === key;
      const matchesSearch = String(sensor.id ?? "")
        .toLowerCase().includes(search.trim().toLowerCase());
      return matchesFilter && matchesSearch && Boolean(state);
    });

    result = [...result].sort((a, b) => {
      if (sort === "trend") return a.trend - b.trend;
      if (sort === "improving") return b.trend - a.trend;
      return a.speed - b.speed;
    });

    return result;
  }, [records, filter, search, sort]);

  const selected = records.find((sensor) => sensor.id === selectedId)
    ?? filtered[0]
    ?? records[0]
    ?? null;

  const shown = showAll ? filtered : filtered.slice(0, 12);

  if (loading) {
    return (
      <section className="page hotspots-page">
        <div className="hotspots-hero">
          <div>
            <span className="hotspots-kicker">HOTSPOT INTELLIGENCE</span>
            <h1>Find pressure before it becomes a problem.</h1>
            <p>Loading the forecast pressure field…</p>
          </div>
        </div>
      </section>
    );
  }

  const filters = [
    { id: "all", label: "All sensors", count: records.length },
    { id: "pressure", label: "Pressure", count: records.filter(s => s.speed < 30).length },
    { id: "moderate", label: "Moderate", count: records.filter(s => s.speed >= 30 && s.speed < 45).length },
    { id: "moving", label: "Moving", count: analysis.moving.length },
    { id: "free", label: "Free flow", count: analysis.freeFlow.length },
  ];

  const selectedState = selected ? trafficState(selected.speed) : null;
  const selectedBehavior = selected ? behaviorFor(selected) : null;

  return (
    <section className="page hotspots-page">
      <header className="hotspots-hero">
        <div className="hotspots-hero__copy">
          <span className="hotspots-kicker"><Gauge size={14} /> Hotspot intelligence</span>
          <h1>Find pressure before it becomes a problem.</h1>
          <p>Explore predicted speeds across the sensor network, identify worsening conditions, and inspect how each location changes over the forecast horizon.</p>
        </div>
        <div className="hotspots-hero__status">
          {analysis.severe.length > 0 ? <AlertTriangle size={18} /> : <ShieldCheck size={18} />}
          <strong>{analysis.severe.length > 0 ? `${analysis.severe.length} severe` : "No severe pressure"}</strong>
          <span>at +{horizon} min</span>
        </div>
      </header>

      <div className="hotspots-summary">
        <div className="hotspots-summary__item is-below45">
          <span>LOCATIONS BELOW 45 MPH</span><strong>{analysis.pressure.length}</strong><small>below 45 mph</small>
        </div>
        <div className="hotspots-summary__item is-moving">
          <span>MOVING</span><strong>{analysis.moving.length}</strong><small>45–55 mph</small>
        </div>
        <div className="hotspots-summary__item is-free">
          <span>FREE FLOW</span><strong>{analysis.freeFlow.length}</strong><small>55 mph or above</small>
        </div>
      </div>

      <div className="hotspots-toolbar">
        <div>
          <span>FORECAST HORIZON</span>
          <b>Explore predictions at +{horizon} minutes</b>
        </div>
        <div className="hotspots-tabs">
          {HORIZONS.map((item) => (
            <button key={item} className={horizon === item ? "is-active" : ""}
              onClick={() => { setHorizon(item); setShowAll(false); }}>
              +{item}m
            </button>
          ))}
        </div>
      </div>

      <div className="uf-sensor-explorer">
        <div className="uf-explorer-heading">
          <div>
            <span className="uf-explorer-eyebrow">NETWORK EXPLORER</span>
            <h2>Explore sensor conditions</h2>
            <p>Select a sensor to inspect its forecast trajectory.</p>
          </div>
          <span className="uf-explorer-total">{filtered.length} matching sensors</span>
        </div>

        <div className="uf-explorer-controls">
          <label className="uf-sensor-search">
            <Search size={16} />
            <input value={search} onChange={(event) => { setSearch(event.target.value); setShowAll(false); }}
              placeholder="Search sensor ID…" aria-label="Search sensor ID" />
            {search && <button type="button" onClick={() => setSearch("")} aria-label="Clear search"><X size={15} /></button>}
          </label>
          <label className="uf-sensor-sort">
            <ArrowUpDown size={15} />
            <select value={sort} onChange={(event) => setSort(event.target.value)} aria-label="Sort sensors">
              <option value="speed">Lowest predicted speed</option>
              <option value="trend">Worsening trend first</option>
              <option value="improving">Improving trend first</option>
            </select>
          </label>
        </div>

        <div className="uf-sensor-filters" aria-label="Filter sensor conditions">
          {filters.map((item) => (
            <button key={item.id} type="button"
              className={filter === item.id ? "is-active" : ""}
              onClick={() => { setFilter(item.id); setShowAll(false); }}>
              {item.label}<span>{item.count}</span>
            </button>
          ))}
        </div>

        <div className="uf-explorer-layout">
          <div className="uf-sensor-results">
            <div className="uf-sensor-grid">
              {shown.map((sensor, index) => {
                const state = trafficState(sensor.speed);
                const behavior = behaviorFor(sensor);
                const active = inspectorOpen && selected?.id === sensor.id;
                const category = sensor.speed < 30 ? "pressure"
                  : sensor.speed < 45 ? "moderate"
                  : sensor.speed < 55 ? "moving" : "free";

                return (
                  <button type="button" key={sensor.id}
                    className={`uf-sensor-card ${active ? "is-selected" : ""} is-${category}`}
                    onClick={() => { setSelectedId(sensor.id); setInspectorOpen(true); }}
                    aria-pressed={active}>
                    <div className="uf-sensor-card__top">
                      <span className="uf-sensor-card__rank">{String(index + 1).padStart(2, "0")}</span>
                      <span className={`uf-sensor-pill is-${category}`}>{state.label}</span>
                      <ChevronRight size={15} className="uf-sensor-card__chevron" />
                    </div>
                    <div className="uf-sensor-card__identity">
                      <strong>{sensor.id}</strong>
                      <span>{sensor.speed.toFixed(1)} <small>mph</small></span>
                    </div>
                    <MiniTrajectory sensor={sensor} />
                    <div className="uf-sensor-card__bottom">
                      <span className={`uf-sensor-trend is-${behavior.tone}`}>
                        {behavior.direction === "down" ? <TrendingDown size={13} />
                          : behavior.direction === "up" ? <TrendingUp size={13} /> : <Minus size={13} />}
                        {behavior.label}
                      </span>
                      <span className="uf-sensor-card__horizon">+{horizon}m</span>
                    </div>
                  </button>
                );
              })}
            </div>

            {filtered.length === 0 && (
              <div className="uf-sensor-empty">
                <Search size={22} />
                <strong>No matching sensors</strong>
                <span>Try another ID or condition filter.</span>
                <button type="button" onClick={() => { setSearch(""); setFilter("all"); }}>Clear filters</button>
              </div>
            )}

            {filtered.length > 12 && (
              <button type="button" className="uf-show-more" onClick={() => setShowAll(!showAll)}>
                {showAll ? "Show fewer sensors" : `Show all ${filtered.length} matching sensors`}
              </button>
            )}
          </div>

          {inspectorOpen && selected && (
            <div
              className="uf-sensor-drawer-backdrop"
              onMouseDown={(event) => {
                if (event.target === event.currentTarget) {
                  setInspectorOpen(false);
                }
              }}
            >
              <aside
                className="uf-sensor-inspector"
                role="dialog"
                aria-modal="true"
                aria-label="Selected sensor details"
              >
                <button
                  type="button"
                  className="uf-sensor-drawer-close"
                  onClick={() => setInspectorOpen(false)}
                  aria-label="Close sensor details"
                >
                  <X size={17} />
                  <span>Close</span>
                </button>
            {selected ? (
              <>
                <div className="uf-inspector-top">
                  <span className="uf-explorer-eyebrow">SELECTED SENSOR</span>
                  <span className={`uf-sensor-pill is-${selected.speed < 30 ? "pressure" : selected.speed < 45 ? "moderate" : selected.speed < 55 ? "moving" : "free"}`}>
                    {selectedState.label}
                  </span>
                </div>
                <h2>{selected.id}</h2>
                <p className="uf-inspector-description">Predicted speed at +{horizon} minutes</p>
                <div className="uf-inspector-speed">
                  <strong>{selected.speed.toFixed(1)}</strong><span>mph</span>
                </div>
                <div className="uf-inspector-trend">
                  {selectedBehavior.direction === "down" ? <TrendingDown size={16} />
                    : selectedBehavior.direction === "up" ? <TrendingUp size={16} /> : <Minus size={16} />}
                  <span>{selectedBehavior.label}</span>
                </div>
                <div className="uf-inspector-divider" />
                <span className="uf-explorer-eyebrow">FORECAST TRAJECTORY</span>
                <div className="uf-inspector-trajectory"><MiniTrajectory sensor={selected} /></div>
                <div className="uf-inspector-horizon-list">
                  {HORIZONS.map((item) => {
                    const value = Number(selected[`speed${item}`] ?? 0);
                    const state = trafficState(value);
                    return (
                      <div key={item}>
                        <span>+{item} min</span>
                        <b>{value.toFixed(1)} mph</b>
                        <i className={`is-${state.tone}`} />
                      </div>
                    );
                  })}
                </div>
                <div className="uf-inspector-note">
                  <MapPinned size={15} />
                  <span>This is a model forecast signal, not a verified incident report.</span>
                </div>
              </>
            ) : (
              <div className="uf-sensor-empty"><Gauge size={22} /><strong>No sensor data</strong></div>
            )}
          </aside>
            </div>
          )}
        </div>
      </div>

      <div className="hotspots-explainer">
        <div className="hotspots-explainer__icon"><MapPinned size={17} /></div>
        <span>HOW TO READ THIS</span>
        <h2>Hotspots are forecast signals, not incidents.</h2>
        <p>A hotspot means UrbanFlow expects lower speed at that location. It does not claim that an accident, closure, or other real-world event is occurring.</p>
        <div className="hotspots-legend">
          <div><i className="is-red" /><span><b>Pressure</b><small>below 30 mph</small></span></div>
          <div><i className="is-amber" /><span><b>Moderate</b><small>30–45 mph</small></span></div>
          <div><i className="is-blue" /><span><b>Moving</b><small>45–55 mph</small></span></div>
          <div><i className="is-green" /><span><b>Free flow</b><small>55+ mph</small></span></div>
        </div>
        <div className="hotspots-explainer__footer"><AlertTriangle size={14} /><span>Ranking uses model-predicted speed, not reported incidents.</span></div>
      </div>
    </section>
  );
}
