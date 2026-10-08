import { useEffect, useMemo, useState } from "react";
import {
  GitBranch,
  Link2,
  MapPin,
  Network as NetworkIcon,
  RadioTower,
  Route,
  Sparkles,
} from "lucide-react";
import { horizonValues, trafficState } from "../data";

const HORIZONS = [15, 30, 45, 60];

function getSensorKey(item, index) {
  return `S-${String(index + 1).padStart(3, "0")}`;
}

function getLat(item) {
  return Number(item?.lat ?? item?.latitude ?? item?.y ?? 0);
}

function getLon(item) {
  return Number(item?.lon ?? item?.lng ?? item?.longitude ?? item?.x ?? 0);
}

function normalizeEdges(payload) {
  const features = Array.isArray(payload?.features)
    ? payload.features
    : Array.isArray(payload)
      ? payload
      : [];

  return features
    .map((feature, index) => {
      const properties = feature?.properties || {};
      const geometry = feature?.geometry;

      if (
        geometry?.type !== "LineString" ||
        !Array.isArray(geometry.coordinates) ||
        geometry.coordinates.length < 2
      ) {
        return null;
      }

      const fromIndex = Number(properties.fromIndex);
      const toIndex = Number(properties.toIndex);

      const coordinates = geometry.coordinates
        .map((coordinate) => ({
          lon: Number(coordinate?.[0]),
          lat: Number(coordinate?.[1]),
        }))
        .filter(
          (coordinate) =>
            Number.isFinite(coordinate.lon) &&
            Number.isFinite(coordinate.lat)
        );

      if (coordinates.length < 2) return null;

      return {
        id: `edge-${index}`,
        sourceIndex: Number.isInteger(fromIndex) ? fromIndex : null,
        targetIndex: Number.isInteger(toIndex) ? toIndex : null,
        fromSensor: properties.from != null ? String(properties.from) : "",
        toSensor: properties.to != null ? String(properties.to) : "",
        weight: Number(properties.weight ?? 1),
        coordinates,
      };
    })
    .filter(Boolean);
}

function normalizeLocations(payload) {
  const raw = Array.isArray(payload)
    ? payload
    : payload?.sensors || payload?.locations || payload?.nodes || [];

  if (!Array.isArray(raw)) return [];

  return raw
    .map((item, index) => ({
      key: getSensorKey(item, index),
      sensorId: String(item?.sensor_id ?? item?.id ?? index),
      lat: getLat(item),
      lon: getLon(item),
      index: Number(item?.index ?? index),
    }))
    .filter(
      (item) =>
        Number.isFinite(item.lat) &&
        Number.isFinite(item.lon)
    );
}

function buildGraph(locations, rawEdges) {
  const sensorIdToKey = new Map(
    locations.map((location) => [location.sensorId, location.key])
  );

  return rawEdges
    .map((edge) => {
      let source =
        edge.sourceIndex != null && locations[edge.sourceIndex]
          ? locations[edge.sourceIndex].key
          : sensorIdToKey.get(edge.fromSensor);

      let target =
        edge.targetIndex != null && locations[edge.targetIndex]
          ? locations[edge.targetIndex].key
          : sensorIdToKey.get(edge.toSensor);

      return {
        ...edge,
        source,
        target,
      };
    })
    .filter(
      (edge) =>
        edge.source &&
        edge.target &&
        edge.source !== edge.target
    );
}

function NetworkGraph({
  locations,
  edges,
  values,
  selectedKey,
  onSelect,
}) {
  const width = 1000;
  const height = 560;
  const padding = 55;

  const [zoom, setZoom] = useState(1);

  const zoomWidth = width / zoom;
  const zoomHeight = height / zoom;
  const zoomX = (width - zoomWidth) / 2;
  const zoomY = (height - zoomHeight) / 2;

  const zoomIn = () => {
    setZoom((value) => Math.min(2.4, Number((value + 0.25).toFixed(2))));
  };

  const zoomOut = () => {
    setZoom((value) => Math.max(0.75, Number((value - 0.25).toFixed(2))));
  };

  const resetZoom = () => setZoom(1);

  const positions = useMemo(() => {
    if (!locations.length) return new Map();

    const lats = locations.map((item) => item.lat);
    const lons = locations.map((item) => item.lon);

    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);

    const latRange = maxLat - minLat || 1;
    const lonRange = maxLon - minLon || 1;

    const result = new Map();

    locations.forEach((item) => {
      result.set(item.key, {
        x:
          padding +
          ((item.lon - minLon) / lonRange) *
            (width - padding * 2),
        y:
          height -
          padding -
          ((item.lat - minLat) / latRange) *
            (height - padding * 2),
      });
    });

    return result;
  }, [locations]);

  const neighborKeys = useMemo(() => {
    if (!selectedKey) return new Set();

    const result = new Set();

    edges.forEach((edge) => {
      if (edge.source === selectedKey) result.add(edge.target);
      if (edge.target === selectedKey) result.add(edge.source);
    });

    return result;
  }, [edges, selectedKey]);

  const selectedIndex = locations.findIndex(
    (location) => location.key === selectedKey
  );

  const selectedSpeed = Number(
    values[selectedIndex >= 0 ? selectedIndex : 0] || 0
  );

  const project = (coordinate) => {
    const lats = locations.map((item) => item.lat);
    const lons = locations.map((item) => item.lon);

    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLon = Math.min(...lons);
    const maxLon = Math.max(...lons);

    const latRange = maxLat - minLat || 1;
    const lonRange = maxLon - minLon || 1;

    return {
      x:
        padding +
        ((coordinate.lon - minLon) / lonRange) *
          (width - padding * 2),
      y:
        height -
        padding -
        ((coordinate.lat - minLat) / latRange) *
          (height - padding * 2),
    };
  };

  return (
    <div className="network-graph">
      <svg
        className="network-graph__svg"
        viewBox={`${zoomX} ${zoomY} ${zoomWidth} ${zoomHeight}`}
        role="img"
        aria-label="UrbanFlow sensor relationship graph"
      >
        <defs>
          <filter id="network-selection-glow">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <rect
          x="0"
          y="0"
          width={width}
          height={height}
          className="network-graph__background"
        />

        <g className="network-graph__grid">
          {Array.from({ length: 11 }).map((_, index) => (
            <line
              key={`v-${index}`}
              x1={index * 100}
              x2={index * 100}
              y1="0"
              y2={height}
            />
          ))}

          {Array.from({ length: 7 }).map((_, index) => (
            <line
              key={`h-${index}`}
              x1="0"
              x2={width}
              y1={index * 93}
              y2={index * 93}
            />
          ))}
        </g>

        <g className="network-graph__edges">
          {edges.map((edge) => {
            const connected =
              edge.source === selectedKey ||
              edge.target === selectedKey;

            const dimmed =
              Boolean(selectedKey) && !connected;

            const path = edge.coordinates
              .map((coordinate, index) => {
                const point = project(coordinate);
                return `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`;
              })
              .join(" ");

            return (
              <g key={edge.id}>
                <path
                  id={`network-edge-path-${edge.id}`}
                  d={path}
                  className={`network-edge ${
                    connected ? "is-connected" : ""
                  } ${dimmed ? "is-dimmed" : ""}`}
                />

                {connected && (
                  <circle
                    r="3.2"
                    className="network-edge__pulse"
                  >
                    <animateMotion
                      dur={`${2.1 + (edge.id.length % 4) * 0.18}s`}
                      begin={`${(edge.id.length % 5) * 0.32}s`}
                      repeatCount="indefinite"
                    >
                      <mpath
                        href={`#network-edge-path-${edge.id}`}
                      />
                    </animateMotion>
                  </circle>
                )}
              </g>
            );
          })}
        </g>

        <g className="network-graph__nodes">
          {locations.map((location, index) => {
            const point = positions.get(location.key);
            if (!point) return null;

            const speed = Number(values[index] || 0);
            const state = trafficState(speed);
            const selected = location.key === selectedKey;
            const neighbor = neighborKeys.has(location.key);
            const dimmed =
              selectedKey &&
              !selected &&
              !neighbor;

            return (
              <g
                key={location.key}
                className={`network-node ${
                  selected ? "is-selected" : ""
                } ${neighbor ? "is-neighbor" : ""} ${
                  dimmed ? "is-dimmed" : ""
                }`}
                transform={`translate(${point.x}, ${point.y})`}
                onClick={() => onSelect(location.key)}
              >
                {selected && (
                  <circle
                    r="16"
                    className="network-node__halo"
                    filter="url(#network-selection-glow)"
                  />
                )}

                {selected && (
                  <circle
                    r="10"
                    className="network-node__selection-ring"
                  />
                )}

                <circle
                  r={selected ? 7 : neighbor ? 5.5 : 4}
                  className={`network-node__dot is-${state.key}`}
                />
              </g>
            );
          })}
        </g>
      </svg>

      <div className="network-graph__zoom" aria-label="Graph zoom controls">
        <button
          type="button"
          onClick={zoomOut}
          disabled={zoom <= 0.75}
          aria-label="Zoom out"
        >
          −
        </button>

        <button
          type="button"
          className="network-graph__zoom-reset"
          onClick={resetZoom}
          aria-label="Reset zoom"
        >
          {Math.round(zoom * 100)}%
        </button>

        <button
          type="button"
          onClick={zoomIn}
          disabled={zoom >= 2.4}
          aria-label="Zoom in"
        >
          +
        </button>
      </div>

      <div className="network-graph__legend">
        <span>
          <i className="is-free" /> Free flow
        </span>
        <span>
          <i className="is-moderate" /> Moderate
        </span>
        <span>
          <i className="is-moving" /> Moving
        </span>
        <span>
          <i className="is-pressure" /> Pressure
        </span>
      </div>

      <div className="network-graph__hint">
        <GitBranch size={14} />
        Select a sensor to inspect its spatial neighborhood
      </div>

      <div className="network-graph__selected">
        <span>SELECTED</span>
        <b>{selectedKey}</b>
        <small>
          {selectedSpeed ? `${selectedSpeed.toFixed(1)} mph` : "—"}
        </small>
      </div>
    </div>
  );
}

export default function Network({ predictions, loading }) {
  const [horizon, setHorizon] = useState(15);
  const [selectedKey, setSelectedKey] = useState("S-001");
  const [locations, setLocations] = useState([]);
  const [rawEdges, setRawEdges] = useState([]);
  const [dataLoading, setDataLoading] = useState(true);
  const [dataError, setDataError] = useState("");

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      fetch("/sensor-locations.json").then((response) => {
        if (!response.ok) {
          throw new Error("Sensor locations could not be loaded.");
        }
        return response.json();
      }),
      fetch("/urbanflow-visual-graph.json").then((response) => {
        if (!response.ok) {
          throw new Error("UrbanFlow graph could not be loaded.");
        }
        return response.json();
      }),
    ])
      .then(([locationData, graphData]) => {
        if (cancelled) return;

        setLocations(normalizeLocations(locationData));
        setRawEdges(normalizeEdges(graphData));
        setDataError("");
      })
      .catch((error) => {
        if (!cancelled) setDataError(error.message);
      })
      .finally(() => {
        if (!cancelled) setDataLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const values = horizonValues(predictions, horizon);

  const graphEdges = useMemo(
    () => buildGraph(locations, rawEdges),
    [locations, rawEdges]
  );

  const selectedIndex = locations.findIndex(
    (location) => location.key === selectedKey
  );

  const selectedSpeed = Number(
    values[selectedIndex >= 0 ? selectedIndex : 0] || 0
  );

  const selectedState = trafficState(selectedSpeed);

  const neighbors = useMemo(() => {
    if (!selectedKey) return [];

    const map = new Map();

    graphEdges.forEach((edge) => {
      if (edge.source === selectedKey) {
        const existing = map.get(edge.target) || 0;
        map.set(edge.target, Math.max(existing, edge.weight));
      }

      if (edge.target === selectedKey) {
        const existing = map.get(edge.source) || 0;
        map.set(edge.source, Math.max(existing, edge.weight));
      }
    });

    return [...map.entries()]
      .map(([key, weight]) => ({
        key,
        weight,
        index: locations.findIndex((location) => location.key === key),
      }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 5);
  }, [graphEdges, locations, selectedKey]);

  const averageSpeed = useMemo(() => {
    const valid = values.filter((value) => Number.isFinite(Number(value)));

    if (!valid.length) return 0;

    return (
      valid.reduce((sum, value) => sum + Number(value), 0) /
      valid.length
    );
  }, [values]);

  // The frontend intentionally renders a filtered visual graph.
  // The ML model itself uses the untouched 207 × 207 adjacency graph:
  // 1,722 directed links, approximately 4.0% density.
  const MODEL_GRAPH_LINKS = 1722;
  const MODEL_GRAPH_DENSITY = "4.0";

  const averageVisibleLinks = locations.length
    ? (graphEdges.length / locations.length).toFixed(1)
    : "0.0";

  const strongestNeighbor = neighbors[0] || null;

  const selectedTrajectory = HORIZONS.map((item) => ({
    horizon: item,
    speed: Number(
      horizonValues(predictions, item)[selectedIndex >= 0 ? selectedIndex : 0] || 0
    ),
  }));

  const trajectoryMin = Math.min(
    ...selectedTrajectory.map((item) => item.speed)
  );

  const trajectoryMax = Math.max(
    ...selectedTrajectory.map((item) => item.speed)
  );

  const trajectoryRange = Math.max(
    trajectoryMax - trajectoryMin,
    1
  );

  if (loading || dataLoading) {
    return (
      <section className="page network-page">
        <div className="network-hero">
          <div>
            <span className="network-kicker">
              <NetworkIcon size={14} /> Spatial intelligence
            </span>
            <h1>Understand the network behind the forecast.</h1>
            <p>Loading UrbanFlow's sensor graph…</p>
          </div>
        </div>
      </section>
    );
  }

  if (dataError) {
    return (
      <section className="page network-page">
        <div className="network-error">
          <RadioTower size={20} />
          <h2>Network data unavailable</h2>
          <p>{dataError}</p>
        </div>
      </section>
    );
  }

  return (
    <section className="page network-page">
      <header className="network-hero">
        <div>
          <span className="network-kicker">
            <NetworkIcon size={14} /> Spatial intelligence
          </span>

          <h1>Understand the network behind the forecast.</h1>

          <p>
            UrbanFlow models traffic as a connected spatial system. Select a
            sensor to see the neighboring relationships that give the GNN its
            spatial context.
          </p>
        </div>

        <div className="network-hero__metrics">
          <div className="network-hero__metric">
            <NetworkIcon size={15} />
            <strong>{locations.length}</strong>
            <span>sensor nodes</span>
          </div>

          <div className="network-hero__metric">
            <Route size={15} />
            <strong>{MODEL_GRAPH_LINKS.toLocaleString()}</strong>
            <span>model graph links</span>
          </div>

          <div className="network-hero__metric">
            <GitBranch size={15} />
            <strong>{MODEL_GRAPH_DENSITY}%</strong>
            <span>model graph density</span>
          </div>
        </div>
      </header>

      <div className="network-controlbar">
        <div className="network-controlbar__title">
          <span>FORECAST LAYER</span>

          <div className="network-horizons">
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

        <div className="network-controlbar__context">
          <div>
            <span>NETWORK VIEW</span>
            <b>{locations.length} sensors</b>
          </div>

          <div>
            <span>VISIBLE LINKS</span>
            <b>{graphEdges.length.toLocaleString()}</b>
          </div>

          <div>
            <span>AVG. VISIBLE DEGREE</span>
            <b>{averageVisibleLinks}</b>
          </div>
        </div>
      </div>

      <div className="network-main">
        <div className="network-graph-card">
          <div className="network-graph-card__head">
            <div>
              <span>SPATIAL GRAPH</span>
              <h2>Sensor relationships</h2>
            </div>

            <div className="network-graph-card__meta">
              <span>
                <i /> {graphEdges.length.toLocaleString()} rendered links
              </span>
              <small>
                {horizon} min traffic layer
              </small>
            </div>
          </div>

          <NetworkGraph
            locations={locations}
            edges={graphEdges}
            values={values}
            selectedKey={selectedKey}
            onSelect={setSelectedKey}
          />
        </div>

        <aside className="network-inspector">
          <div className="network-inspector__top">
            <div className="network-inspector__icon">
              <MapPin size={18} />
            </div>

            <span>SELECTED SENSOR</span>
          </div>

          <div className="network-inspector__identity">
            <h2>{selectedKey}</h2>

            <span className={`network-state is-${selectedState.tone}`}>
              {selectedState.label}
            </span>
          </div>

          <div className="network-inspector__speed">
            <strong>{selectedSpeed.toFixed(1)}</strong>
            <span>mph at +{horizon} min</span>
          </div>

          <div className="network-trajectory">
            <div className="network-trajectory__head">
              <span>FORECAST TRAJECTORY</span>
              <b>
                {selectedTrajectory[0]?.speed.toFixed(1)} →{" "}
                {selectedTrajectory[selectedTrajectory.length - 1]?.speed.toFixed(1)} mph
              </b>
            </div>

            <div className="network-trajectory__bars">
              {selectedTrajectory.map((item) => {
                const height =
                  30 +
                  ((item.speed - trajectoryMin) / trajectoryRange) * 55;

                return (
                  <div key={item.horizon} className="network-trajectory__item">
                    <div className="network-trajectory__bar">
                      <i style={{ height: `${height}%` }} />
                    </div>
                    <b>{item.speed.toFixed(1)}</b>
                    <span>+{item.horizon}m</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="network-inspector__section">
            <span className="network-section-label">
              <Link2 size={14} /> NEARBY SENSORS
            </span>
          </div>

          <div className="network-neighbors">
            {neighbors.length ? (
              neighbors.map((neighbor, index) => {
                const neighborSpeed = Number(
                  values[neighbor.index] || 0
                );
                const neighborState = trafficState(neighborSpeed);

                return (
                  <button
                    key={neighbor.key}
                    onClick={() => setSelectedKey(neighbor.key)}
                  >
                    <span className="network-neighbor-rank">
                      0{index + 1}
                    </span>

                    <span className="network-neighbor-id">
                      <b>{neighbor.key}</b>
                      <small>
                        {neighbor.weight.toFixed(2)} relationship
                      </small>
                    </span>

                    <span
                      className={`network-neighbor-speed is-${neighborState.tone}`}
                    >
                      {neighborSpeed.toFixed(1)}
                    </span>
                  </button>
                );
              })
            ) : (
              <div className="network-neighbors__empty">
                Select another node to inspect its local relationships.
              </div>
            )}
          </div><div className="network-inspector__footer">
            <Sparkles size={15} />
            <span>
              The GNN uses surrounding sensor relationships as spatial context
              before producing the forecast.
            </span>
          </div>
        </aside>
      </div>

      <div className="network-structure">
        <div className="network-structure__intro">
          <span>NETWORK STRUCTURE</span>
          <h2>The spatial graph behind UrbanFlow</h2>
          <p>
            The network is not just a collection of independent sensors.
            Connections let the model reason about nearby traffic conditions.
          </p>
        </div>

        <div className="network-structure__stats">
          <div>
            <NetworkIcon size={18} />
            <strong>{locations.length}</strong>
            <span>sensor nodes</span>
          </div>

          <div>
            <Route size={18} />
            <strong>{MODEL_GRAPH_LINKS.toLocaleString()}</strong>
            <span>model graph links</span>
          </div>

          <div>
            <GitBranch size={18} />
            <strong>{MODEL_GRAPH_DENSITY}%</strong>
            <span>model graph density</span>
          </div>

          <div>
            <Link2 size={18} />
            <strong>{averageVisibleLinks}</strong>
            <span>avg. visible links / node</span>
          </div>
        </div>
      </div>
    </section>
  );
}
