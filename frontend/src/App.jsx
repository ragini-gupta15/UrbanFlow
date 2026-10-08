import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity, Bell, BrainCircuit, CloudSun, Gauge, GitBranch,
  LayoutDashboard, Menu, RefreshCw, TrafficCone, X, Zap
} from "lucide-react";
import { API_URL, average, buildSensorRecords, horizonValues, trafficState } from "./data";
import UrbanMap from "./components/UrbanMap";
import Forecasts from "./pages/Forecasts";
import Network from "./pages/Network";
import Hotspots from "./pages/Hotspots";
import ModelIntelligence from "./pages/ModelIntelligence";
import "./App.css";

const navigation = [
  ["Command Center", LayoutDashboard],
  ["Forecasts", Activity],
  ["Network", GitBranch],
  ["Hotspots", Gauge],
  ["Model Intelligence", BrainCircuit],
];

export default function App() {
  const [activePage, setActivePage] = useState("Command Center");
  const [predictions, setPredictions] = useState(null);
  const [weather, setWeather] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadLiveData = (showLoader = false) => {
    if (showLoader) setRefreshing(true);
    setError("");
    return Promise.all([
      fetch(`${API_URL}/demo-predict`).then((response) => {
        if (!response.ok) throw new Error(`Prediction request failed (${response.status})`);
        return response.json();
      }),
      fetch(`${API_URL}/weather-data`).then((response) => {
        if (!response.ok) throw new Error(`Weather request failed (${response.status})`);
        return response.json();
      }),
    ])
      .then(([predictionData, weatherData]) => {
        setPredictions(predictionData);
        setWeather(weatherData);
        setLastUpdated(new Date());
      })
      .catch((err) => setError(err.message))
      .finally(() => {
        setLoading(false);
        setRefreshing(false);
      });
  };

  useEffect(() => {
    loadLiveData(true);
  }, []);

  const sensors = useMemo(() => buildSensorRecords(predictions || {}), [predictions]);

  const navigate = (page) => {
    setActivePage(page);
    setMobileOpen(false);
  };

  return (
    <div className="urbanflow-app">
      <Sidebar activePage={activePage} navigate={navigate} />
      <AnimatePresence>
        {mobileOpen && (
          <motion.div className="mobile-nav" initial={{ x: "-100%" }} animate={{ x: 0 }} exit={{ x: "-100%" }}>
            <button className="mobile-nav__close" onClick={() => setMobileOpen(false)}><X /></button>
            <Brand />
            <nav>{navigation.map(([name, Icon]) => <NavItem key={name} name={name} Icon={Icon} active={activePage === name} onClick={() => navigate(name)} />)}</nav>
          </motion.div>
        )}
      </AnimatePresence>
      <main className="app-main">
        <header className="topbar">
          <div className="topbar__left">
            <button className="mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={19} /></button>
            <div>
              <div className="topbar__crumb">URBANFLOW / {activePage.toUpperCase()}</div>
              <div className="topbar__title">Predictive mobility intelligence</div>
            </div>
          </div>
          <div className="topbar__right">
            <div className={`engine-pill ${loading ? "is-loading" : error ? "is-error" : ""}`}>
              <span /> {loading ? "SYNCING ENGINE" : error ? "ENGINE OFFLINE" : "INFERENCE ENGINE ONLINE"}
            </div>
            <div className="last-updated">{lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "Waiting for data"}</div>
            <button className={`icon-button refresh-button ${refreshing ? "is-refreshing" : ""}`} onClick={() => loadLiveData(true)} aria-label="Refresh live UrbanFlow data" title="Refresh live data"><RefreshCw size={16} /></button>
          </div>
        </header>

        {error && <div className="global-error"><TrafficCone size={17} /><div><b>Prediction service unavailable</b><span>{error}. Start the FastAPI service on port 8000.</span></div></div>}

        {activePage === "Command Center" && <CommandCenter predictions={predictions} weather={weather} sensors={sensors} loading={loading} navigate={navigate} />}
        {activePage === "Forecasts" && <Forecasts predictions={predictions} loading={loading} />}
        {activePage === "Network" && <Network predictions={predictions} loading={loading} />}
        {activePage === "Hotspots" && <Hotspots sensors={sensors} loading={loading} />}
        {activePage === "Model Intelligence" && <ModelIntelligence predictions={predictions} weather={weather} loading={loading} />}
      </main>
    </div>
  );
}

function Brand() {
  return <div className="brand"><div className="brand__mark"><Zap size={18} /></div><div><b>UrbanFlow</b><span>Predictive mobility</span></div></div>;
}

function Sidebar({ activePage, navigate }) {
  return <aside className="sidebar"><Brand /><div className="sidebar__label">Explore the city</div><nav>{navigation.map(([name, Icon]) => <NavItem key={name} name={name} Icon={Icon} active={activePage === name} onClick={() => navigate(name)} />)}</nav><div className="sidebar__bottom"><div className="engine-card"><div className="engine-card__top"><span><i /> Engine</span><b>LIVE</b></div><p>207 sensors · 4 horizons</p><div className="engine-card__line"><span /><span /><span /><span /></div></div><small>UrbanFlow AI · Final model</small></div></aside>;
}

function NavItem({ name, Icon, active, onClick }) {
  return <button className={`nav-item ${active ? "is-active" : ""}`} onClick={onClick}><Icon size={17} /><span>{name}</span>{active && <motion.i layoutId="nav-dot" />}</button>;
}

function CommandCenter({ predictions, weather, sensors, loading, navigate }) {
  const [horizon, setHorizon] = useState(15);
  const [selected, setSelected] = useState(0);

  const values = horizonValues(predictions, horizon);
  const avg = average(values);
  const next = average(horizonValues(predictions, Math.min(60, horizon + 15)));
  const trend = avg ? ((next - avg) / avg) * 100 : 0;
  const selectedSensor = sensors[selected];
  const below45Count = values.filter((value) => value < 45).length;
  const state = trafficState(selectedSensor?.speed15 || 0);

  if (loading) return <LoadingScreen />;
  if (!predictions) return null;

  return (
    <section className="page command-center command-center--operations">

      <div className="operations-header">
        <div>
          <div className="live-kicker">
            <span />
            Live urban network
          </div>
          <h1>City traffic intelligence</h1>
          <p>
            Explore how traffic is expected to move across the monitored road
            network.
          </p>
        </div>

        <div className="operations-controls">
          <div className="operations-status">
            <span />
            LIVE MODEL
          </div>

          <div className="operations-horizons">
            {[15, 30, 45, 60].map((item) => (
              <button
                key={item}
                className={horizon === item ? "is-active" : ""}
                onClick={() => setHorizon(item)}
              >
                {item}<small>m</small>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="operations-map-shell">

        <div className="operations-map-top">
          <div className="map-context">
            <span className="map-context__live">
              <i />
              FORECAST ACTIVE
            </span>
            <strong>{horizon}-minute traffic outlook</strong>
            <span>
              {below45Count} locations below 45 mph · 207 monitored locations
            </span>
          </div>

          <div className="map-metrics">
            <div>
              <span>NETWORK SPEED</span>
              <b>{avg.toFixed(1)} <small>mph</small></b>
            </div>
            <div>
              <span>NEXT HORIZON</span>
              <b className={trend > 1 ? "is-negative" : "is-positive"}>
                {trend > 0 ? "↓" : "↑"} {Math.abs(trend).toFixed(1)}%
              </b>
            </div>
          </div>
        </div>

        <div className="operations-map">
          <UrbanMap
            values={values}
          horizon={horizon}
          baselineValues={horizonValues(predictions, 15)}
            selected={selected}
            onSelect={setSelected}
          />

          <div className="map-overlay map-overlay--top-left">
            <span>URBANFLOW</span>
            <b>NETWORK VIEW</b>
          </div>

          <div className="map-overlay map-overlay--bottom-left">
            <span>FORECAST</span>
            <b>{horizon} MIN</b>
          </div>

          <div className="map-overlay map-overlay--bottom-right">
            <span>MONITORED</span>
            <b>207 SENSORS</b>
          </div>
        </div>

        <div className="operations-map-footer">
          <div className="traffic-key">
            <span><i className="is-free" /> Free flow</span>
            <span><i className="is-moving" /> Moving</span>
            <span><i className="is-moderate" /> Moderate</span>
            <span><i className="is-pressure" /> Pressure</span>
          </div>

          <button onClick={() => navigate("Network")}>
            Explore network
            <GitBranch size={14} />
          </button>
        </div>
      </div>

      <div className="operations-lower">

        <section className="operations-panel operations-panel--selected">
          <div className="operations-panel__head">
            <div>
              <span>SENSOR OUTLOOK</span>
              <h2>{selectedSensor?.id || "S-001"}</h2>
            </div>

            <span className={`state-chip state-chip--${state.tone}`}>
              {state.label}
            </span>
          </div>

          <div className="selected-speed">
            <strong>
              {Number(selectedSensor?.[`speed${horizon}`] || 0).toFixed(1)}
            </strong>
            <span>mph expected at +{horizon} min</span>
          </div>

          <div className="selected-forecast">
            {[15, 30, 45, 60].map((item) => (
              <button
                key={item}
                className={horizon === item ? "is-active" : ""}
                onClick={() => setHorizon(item)}
              >
                <span>{item} min</span>
                <b>{selectedSensor?.[`speed${item}`]?.toFixed(1)} mph</b>
              </button>
            ))}
          </div>

          <button
            className="operations-link"
            onClick={() => navigate("Forecasts")}
          >
            View sensor trajectory
            <span>→</span>
          </button>
        </section>

        <section className="operations-panel operations-panel--network">
          <div className="operations-panel__head">
            <div>
              <span>CITY OUTLOOK</span>
              <h2>
                {trend > 1
                  ? "Traffic pressure is rising"
                  : trend < -1
                    ? "Traffic is improving"
                    : "Traffic remains stable"}
              </h2>
            </div>
            <Activity size={18} />
          </div>

          <div className="network-outlook-summary">
            <span>
              {trend > 1
                ? "Network pressure is increasing"
                : trend < -1
                  ? "Network conditions are improving"
                  : "Low change expected"}
            </span>
          </div>

          <div className="network-condition">
            <div>
              <span>NOW</span>
              <b>{avg.toFixed(1)} mph</b>
            </div>
            <div className="condition-arrow">→</div>
            <div>
              <span>NEXT</span>
              <b>{next.toFixed(1)} mph</b>
            </div>
          </div>

          <button
            className="operations-link"
            onClick={() => navigate("Model Intelligence")}
          >
            Explore model insight
            <span>→</span>
          </button>
        </section>

        <section className="operations-panel operations-panel--weather">
          <div className="operations-panel__head">
            <div>
              <span>ENVIRONMENTAL CONTEXT</span>
              <h2>{weather ? `${weather.temperature.toFixed(1)}°` : "—"}</h2>
              <p className="weather-context-subtitle">
                Conditions feeding the forecast
              </p>
            </div>
            <CloudSun size={18} />
          </div>

          <div className="weather-brief">
            <div>
              <span>Humidity</span>
              <b>{weather ? `${weather.humidity.toFixed(0)}%` : "—"}</b>
            </div>
            <div>
              <span>Wind</span>
              <b>{weather ? `${weather.wind_speed.toFixed(1)}` : "—"}</b>
            </div>
          </div>

          <div className="weather-feature-strip">
            <span>Precipitation</span>
            <span>Pressure</span>
            <span>Cloud cover</span>
          </div>

          <button
            className="operations-link"
            onClick={() => navigate("Model Intelligence")}
          >
            6 weather features in model
            <span>→</span>
          </button>
        </section>

      </div>
    </section>
  );
}

function LoadingScreen() { return <section className="page loading-screen"><div className="loading-orbit"><span /><span /><span /></div><h2>Connecting to the city</h2><p>Loading the final UrbanFlow prediction engine…</p></section>; }
