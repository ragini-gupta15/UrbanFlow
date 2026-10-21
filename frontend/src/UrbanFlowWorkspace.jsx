import { useEffect, useMemo, useState } from "react";
import {
  Activity, BrainCircuit, CloudSun, GitBranch, Gauge,
  LayoutDashboard, RefreshCw, Route, Menu, X, ArrowUpRight
} from "lucide-react";
import { API_URL, average, buildSensorRecords, horizonValues, trafficState } from "./data";
import UrbanMap from "./components/UrbanMap";
import Forecasts from "./pages/Forecasts";
import Network from "./pages/Network";
import Hotspots from "./pages/Hotspots";
import ModelIntelligence from "./pages/ModelIntelligence";
import "./App.css";
import "./urbanflow-workspace.css";

const NAV = [
  { name: "Command Center", icon: LayoutDashboard, code: "01" },
  { name: "Forecasts", icon: Activity, code: "02" },
  { name: "Network", icon: GitBranch, code: "03" },
  { name: "Hotspots", icon: Gauge, code: "04" },
  { name: "Model Intelligence", icon: BrainCircuit, code: "05" },
];

export default function UrbanFlowWorkspace() {
  const [page, setPage] = useState("Command Center");
  const [predictions, setPredictions] = useState(null);
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [sampleInput, setSampleInput] = useState("0");
  const [error, setError] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);

  
  
  
async function loadReplay(showRefresh = false, selectedIndex = null) {
  if (showRefresh) setRefreshing(true);
  setError("");

  const requestedIndex = selectedIndex !== null
    ? Math.max(0, Math.min(6837, Number(selectedIndex) || 0))
    : showRefresh
      ? (sampleIndex + 1) % 6838
      : sampleIndex;

  try {
    const response = await fetch(
      `${API_URL}/demo-replay?sample_index=${requestedIndex}`
    );

    if (!response.ok) {
      throw new Error(`Historical replay request failed (${response.status})`);
    }

    const replay = await response.json();
    setPredictions(replay.predictions);
    setWeather(replay.weather);
    setSampleIndex(replay.sample_index);
    setSampleInput(String(replay.sample_index));
    setError("");
  } catch {
    try {
      const response = await fetch("/demo-replay-samples.json");

      if (!response.ok) {
        throw new Error("Saved historical samples could not be loaded.");
      }

      const data = await response.json();
      const samples = data.samples;

      if (!Array.isArray(samples) || samples.length === 0) {
        throw new Error("No saved historical samples are available.");
      }

      // Match the requested index to a saved historical sample.
            // Cycle through the saved samples in order.
      let position;

      if (selectedIndex !== null) {
        const requested = Number(selectedIndex);
        position = samples.findIndex(
          sample => sample.sample_index === requested
        );

        // If the requested index is not saved, choose the nearest sample.
        if (position < 0) {
          position = samples.reduce((best, sample, i) =>
            Math.abs(sample.sample_index - requested) <
            Math.abs(samples[best].sample_index - requested)
              ? i
              : best, 0);
        }
      } else if (showRefresh) {
        const currentPosition = samples.findIndex(
          sample => sample.sample_index === sampleIndex
        );
        position = (currentPosition + 1) % samples.length;
      } else {
        position = samples.findIndex(
          sample => sample.sample_index === sampleIndex
        );
        if (position < 0) position = 0;
      }

      const replay = samples[position];

      setPredictions(replay.predictions);
      setWeather(replay.weather);
      setSampleIndex(replay.sample_index);
      setSampleInput(String(replay.sample_index));
      setError(
        "Historical Demo Mode: showing saved forecasts. Live API data is unavailable."
      );
    } catch (fallbackError) {
      setError(
        fallbackError.message ||
        "Unable to load historical demonstration data."
      );
    }
  } finally {
    setLoading(false);
    setRefreshing(false);
  }
}

  


  useEffect(() => { loadReplay(); }, []);

  const sensors = useMemo(
    () => buildSensorRecords(predictions || {}),
    [predictions]
  );

  function navigate(name) {
    setPage(name);
    setMobileOpen(false);
  }

  return (
    <div className="uf-workspace">
      <aside className={`uf-rail ${mobileOpen ? "uf-rail--open" : ""}`}>
        <div className="uf-brand">
          <div className="uf-brand-mark"><Route size={21} /></div>
          <div><strong>urbanflow<span>.</span></strong><small>MOBILITY INTELLIGENCE</small></div>
          <button className="uf-mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close navigation"><X size={18} /></button>
        </div>

        <div className="uf-rail-label">WORKSPACE</div>
        <nav className="uf-nav">
          {NAV.map(({ name, icon: Icon, code }) => (
            <button key={name} className={`uf-nav-item ${page === name ? "is-active" : ""}`} onClick={() => navigate(name)}>
              <Icon size={18} strokeWidth={1.8} />
              <span>{name}</span>
              <small>{code}</small>
            </button>
          ))}
        </nav>

        <div className="uf-rail-bottom">
          <div className="uf-rail-status">
            <span className="uf-status-dot" />
            <div><strong>Historical replay</strong><small>METR-LA · 207 sensors</small></div>
          </div>
          <div className="uf-rail-foot">SPATIOTEMPORAL FORECASTING<br />GNN + TRANSFORMER + WEATHER</div>
        </div>
      </aside>

      {mobileOpen && <button className="uf-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}

      <main className="uf-main">
        <header className="uf-topbar">
          <div className="uf-topbar-left">
            <button className="uf-menu-button" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={19} /></button>
            <div className="uf-breadcrumb"><span>URBANFLOW</span><b>/</b>{page.toUpperCase()}</div>
          </div>
          <div className="uf-topbar-right">
            <span className="uf-replay-pill"><i /> HISTORICAL REPLAY · {sampleIndex}</span>
            <form className="uf-sample-picker" onSubmit={(event) => {
              event.preventDefault();
              loadReplay(false, sampleInput);
            }}>
              <label htmlFor="uf-sample-index">Sample</label>
              <input
                id="uf-sample-index"
                type="number"
                min="0"
                max="6837"
                value={sampleInput}
                onChange={(event) => setSampleInput(event.target.value)}
                aria-label="Historical replay sample index"
              />
              <button type="submit" disabled={refreshing}>Go</button>
            </form>
            <button className={`uf-refresh ${refreshing ? "is-refreshing" : ""}`} onClick={() => loadReplay(true)} disabled={refreshing} aria-label="Load next historical sample" title="Load next replay sample">
              <RefreshCw size={16} />
              <span>{refreshing ? "Loading" : "Next sample"}</span>
            </button>
          </div>
        </header>

        {error && (
  <div className="uf-error">
    <strong>Historical Demo Mode</strong>
    <span>
      Showing saved historical traffic predictions. Live API data is currently unavailable.
    </span>
    <button type="button" onClick={() => loadReplay(true)}>
      Retry live API
    </button>
  </div>
)}
        {page === "Command Center" && (
          <CommandCenter predictions={predictions} weather={weather} sensors={sensors} loading={loading} navigate={navigate} />
        )}
        {page === "Forecasts" && <Forecasts predictions={predictions} loading={loading} />}
        {page === "Network" && <Network predictions={predictions} loading={loading} />}
        {page === "Hotspots" && <Hotspots sensors={sensors} loading={loading} />}
        {page === "Model Intelligence" && <ModelIntelligence predictions={predictions} weather={weather} loading={loading} />}
      </main>
    </div>
  );
}

function CommandCenter({ predictions, weather, sensors, loading, navigate }) {
  const [horizon, setHorizon] = useState(15);
  const [selected, setSelected] = useState(0);
  const [trafficFilter, setTrafficFilter] = useState("all");
  const values = horizonValues(predictions || {}, horizon);
  const baseline = horizonValues(predictions || {}, 15);
  const networkAverage = average(values);
  const nextAverage = average(horizonValues(predictions || {}, Math.min(60, horizon + 15)));
  const selectedSensor = sensors[selected];
  const state = trafficState(selectedSensor?.[`speed${horizon}`] ?? 0);
  const below45 = values.filter(v => v < 45).length;
  const filterLabels = {
    all: "All sensors",
    congested: "Congested",
    moderate: "Moderate",
    moving: "Moving",
    free: "Free flow",
  };
  const visibleSensorCount = values.filter((rawSpeed) => {
    const speed = Number(rawSpeed);
    if (trafficFilter === "congested") return speed < 30;
    if (trafficFilter === "moderate") return speed >= 30 && speed < 45;
    if (trafficFilter === "moving") return speed >= 45 && speed < 55;
    if (trafficFilter === "free") return speed >= 55;
    return true;
  }).length;

  const matchesTrafficFilter = (rawSpeed, filter) => {
    const speed = Number(rawSpeed);
    if (filter === "congested") return speed < 30;
    if (filter === "moderate") return speed >= 30 && speed < 45;
    if (filter === "moving") return speed >= 45 && speed < 55;
    if (filter === "free") return speed >= 55;
    return true;
  };

  const handleTrafficFilterChange = (nextFilter) => {
    setTrafficFilter(nextFilter);

    const currentSpeed = values[selected];
    if (matchesTrafficFilter(currentSpeed, nextFilter)) return;

    const matchingIndex = values.findIndex((speed) =>
      matchesTrafficFilter(speed, nextFilter)
    );

    if (matchingIndex >= 0) setSelected(matchingIndex);
  };



  if (loading) return <section className="uf-content"><div className="uf-loading"><div className="uf-loader" /><span>Loading historical test replay…</span></div></section>;
  if (!predictions) return <section className="uf-content"><h1>Forecast unavailable</h1><p>Reload the sample after checking the API.</p></section>;

  return (
    <section className="uf-content">
      <div className="uf-page-heading">
        <div>
          <div className="uf-eyebrow"><span /> SPATIAL OPERATIONS / LOS ANGELES</div>
          <h1>City traffic, <em>in context.</em></h1>
          <p>Explore predicted traffic conditions across the METR-LA sensor network.</p>
        </div>
        <div className="uf-heading-note"><span>MODEL OUTPUT</span><strong>4 forecast horizons</strong><small>15 · 30 · 45 · 60 minutes</small></div>
      </div>

      <div className="uf-stat-row">
        <button type="button" className="uf-stat uf-stat--interactive" onClick={() => navigate("Network")} title="Explore the sensor network"><span>MONITORED SENSORS</span><strong>{sensors.length}</strong><small>Explore network ↗</small></button>
        <button type="button" className="uf-stat uf-stat--interactive" onClick={() => navigate("Forecasts")} title="Explore forecast analysis"><span>NETWORK MEAN SPEED</span><strong>{networkAverage.toFixed(1)} <small>mph</small></strong><small>Forecast analysis ↗</small></button>
        <button type="button" className="uf-stat uf-stat--interactive" onClick={() => navigate("Hotspots")} title="Explore traffic hotspots"><span>BELOW 45 MPH</span><strong>{below45}<small> / {values.length}</small></strong><small>Explore hotspots ↗</small></button>
        <button type="button" className="uf-stat uf-stat--interactive" onClick={() => navigate("Model Intelligence")} title="Explore weather features"><span>ENVIRONMENT</span><strong>{weather ? `${Number(weather.temperature).toFixed(1)}°` : "—"} <small>C</small></strong><small>Weather context ↗</small></button>
      </div>

      <section className="uf-map-panel">
        <div className="uf-map-heading">
          <div><div className="uf-section-label">NETWORK EXPLORER <span>·</span> HISTORICAL REPLAY</div><h2>Predicted road conditions</h2><p>Choose a horizon or select a sensor on the map.</p></div>
          <div className="uf-horizon-control" aria-label="Forecast horizon">
            {[15, 30, 45, 60].map(h => <button key={h} className={horizon === h ? "is-active" : ""} onClick={() => setHorizon(h)}>{h}<small>m</small></button>)}
          </div>
        </div>

        <div className="uf-map-filters" role="group" aria-label="Filter map sensors by predicted traffic state">
          <span>SHOW</span>
          {[
            { id: "all", label: "All sensors" },
            { id: "congested", label: "Congested" },
            { id: "moderate", label: "Moderate" },
            { id: "moving", label: "Moving" },
            { id: "free", label: "Free flow" },
          ].map((item) => (
            <button
              type="button"
              key={item.id}
              className={`uf-map-filter ${trafficFilter === item.id ? "is-active" : ""} is-${item.id}`}
              aria-pressed={trafficFilter === item.id}
              onClick={() => handleTrafficFilterChange(item.id)}
            >
              <i />
              {item.label}
            </button>
          ))}
          {trafficFilter !== "all" && (
            <button type="button" className="uf-map-filter-reset" onClick={() => handleTrafficFilterChange("all")}>
              Reset
            </button>
          )}
        </div>

        {visibleSensorCount === 0 && trafficFilter !== "all" && (
          <p className="uf-empty-filter-message" role="status">
            No sensors match “{filterLabels[trafficFilter]}” at the {horizon}-minute forecast horizon.
            Try another traffic filter or select All sensors.
          </p>
        )}

        <div className="uf-map-insight" aria-live="polite">
          <div className="uf-map-insight-status">
            <span className={`uf-map-insight-dot is-${trafficFilter}`} />
            <div>
              <span className="uf-map-insight-label">MAP FILTER</span>
              <strong>{filterLabels[trafficFilter] || "All sensors"}</strong>
            </div>
          </div>
          <div className="uf-map-insight-divider" />
          <div className="uf-map-insight-count">
            <span className="uf-map-insight-label">VISIBLE SENSORS</span>
            <strong>{visibleSensorCount}<small> / {sensors.length}</small></strong>
          </div>
          <div className="uf-map-insight-divider" />
          <div className="uf-map-insight-horizon">
            <span className="uf-map-insight-label">FORECAST HORIZON</span>
            <strong>+{horizon} <small>minutes</small></strong>
          </div>
        </div>

        <div className="uf-map-frame">
          <UrbanMap values={values} baselineValues={baseline} horizon={horizon} selected={selected} onSelect={setSelected} trafficFilter={trafficFilter} />
          <div className="uf-map-tag"><span>LA</span><div><strong>METR-LA</strong><small>Sensor network</small></div></div>
          <div className="uf-map-horizon">T + {horizon} MIN</div>
        </div>
        <div className="uf-map-legend">
          <span><i className="uf-legend-green" /> Free flow</span>
          <span><i className="uf-legend-cyan" /> Moving</span>
          <span><i className="uf-legend-amber" /> Moderate</span>
          <span><i className="uf-legend-red" /> Congested</span>
          <button onClick={() => navigate("Network")}>Explore network <ArrowUpRight size={15} /></button>
        </div>
      </section>

      <div className="uf-detail-grid">
        <section className="uf-detail-card uf-sensor-card">
          <div className="uf-card-top"><div><span className="uf-section-label">SELECTED SENSOR</span><h2>{selectedSensor?.id || `Sensor ${selected + 1}`}</h2></div><span className={`uf-state uf-state--${state.tone}`}>{state.label}</span></div>
          <div className="uf-sensor-speed"><strong>{Number(selectedSensor?.[`speed${horizon}`] ?? 0).toFixed(1)}</strong><span>mph <small>at +{horizon} min</small></span></div>
          <div className="uf-sensor-horizons">{[15, 30, 45, 60].map(h => <button key={h} onClick={() => setHorizon(h)} className={horizon === h ? "is-active" : ""}><span>+{h}m</span><strong>{Number(selectedSensor?.[`speed${h}`] ?? 0).toFixed(1)}</strong></button>)}</div>
          <button className="uf-text-link" onClick={() => navigate("Forecasts")}>View forecast analysis <ArrowUpRight size={15} /></button>
        </section>

        <section className="uf-detail-card">
          <div className="uf-card-top"><div><span className="uf-section-label">NETWORK OUTLOOK</span><h2>Across the next horizon</h2></div><Activity size={18} /></div>
          <div className="uf-compare"><div><span>SELECTED</span><strong>{networkAverage.toFixed(1)} <small>mph</small></strong></div><div className="uf-compare-arrow">→</div><div><span>NEXT</span><strong>{nextAverage.toFixed(1)} <small>mph</small></strong></div></div>
          <p className="uf-muted">Mean predicted speed; the next horizon is capped at 60 minutes.</p>
          <button className="uf-text-link" onClick={() => navigate("Model Intelligence")}>Explore model intelligence <ArrowUpRight size={15} /></button>
        </section>

        <section className="uf-detail-card">
          <div className="uf-card-top"><div><span className="uf-section-label">WEATHER CONTEXT</span><h2>{weather ? `${Number(weather.temperature).toFixed(1)}°C` : "Unavailable"}</h2></div><CloudSun size={19} /></div>
          <div className="uf-weather-grid"><div><span>Humidity</span><strong>{weather ? `${Number(weather.humidity).toFixed(0)}%` : "—"}</strong></div><div><span>Wind speed</span><strong>{weather ? Number(weather.wind_speed).toFixed(1) : "—"}</strong></div><div><span>Precipitation</span><strong>{weather ? Number(weather.precipitation).toFixed(1) : "—"}</strong></div><div><span>Cloud cover</span><strong>{weather ? `${Number(weather.cloud_cover).toFixed(0)}%` : "—"}</strong></div></div>
          <button className="uf-text-link" onClick={() => navigate("Model Intelligence")}>How weather is used <ArrowUpRight size={15} /></button>
        </section>
      </div>
      <footer className="uf-page-foot"><span>URBANFLOW / SPATIOTEMPORAL FORECASTING</span><span>Historical test sample · Not live traffic</span></footer>
    </section>
  );
}
