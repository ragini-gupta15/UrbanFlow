import * as maplibregl from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";

maplibregl.setWorkerUrl(workerUrl);
import { useEffect, useRef } from "react";
import "maplibre-gl/dist/maplibre-gl.css";


const LOS_ANGELES_CENTER = [-118.2437, 34.0522];

const speedColor = [
  "interpolate", ["linear"], ["get", "signal"],
  0, "#ef4444",
  0.25, "#f59e0b",
  0.5, "#18b6d9",
  0.75, "#22c55e",
  1, "#18d6b7"
];

const forecastColor = [
  "interpolate",
  ["linear"],
  ["get", "signal"],
  -15, "#ef4444",
  -5, "#f59e0b",
  0, "#18b6d9",
  5, "#22c55e",
  15, "#18d6b7",
];

const forecastWidth = [
  "interpolate",
  ["linear"],
  ["get", "signal"],
  -15, 4.8,
  -5, 4.0,
  0, 2.4,
  5, 3.5,
  15, 4.8,
];

const forecastOpacity = [
  "interpolate",
  ["linear"],
  ["get", "signal"],
  -15, 0.95,
  -5, 0.9,
  0, 0.72,
  5, 0.9,
  15, 0.95,
];

export default function UrbanMap({
  values = [],
  baselineValues = [],
  horizon = 15,
  selected = 0,
  onSelect = () => {},
}) {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);

  const sensorDataRef = useRef(null);
  const graphDataRef = useRef(null);
  const onSelectRef = useRef(onSelect);

  const forecastRef = useRef({
    values,
    baselineValues,
    horizon,
    selected,
  });

  const applyForecastRef = useRef(null);

  onSelectRef.current = onSelect;

  forecastRef.current = {
    values,
    baselineValues,
    horizon,
    selected,
  };

  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: "https://tiles.openfreemap.org/styles/liberty",
      center: LOS_ANGELES_CENTER,
      zoom: 10.5,
      pitch: 42,
      bearing: -8,
      attributionControl: true,
    });

    mapRef.current = map;

    const applyForecast = () => {
      const currentMap = mapRef.current;
      const sensors = sensorDataRef.current;
      const graph = graphDataRef.current;
      const {
        values: currentValues,
        baselineValues: currentBaseline,
        horizon: currentHorizon,
        selected: currentSelected,
      } = forecastRef.current;

      if (!currentMap || !sensors || !graph) return;

      const sensorSource = currentMap.getSource("urbanflow-sensors");
      const graphSource = currentMap.getSource("urbanflow-graph");

      if (!sensorSource || !graphSource) return;

      const speedByIndex = new Map(
        sensors.map((sensor) => [
          sensor.index,
          Number(currentValues[sensor.index] ?? 0),
        ])
      );

      const baselineByIndex = new Map(
        sensors.map((sensor) => [
          sensor.index,
          Number(currentBaseline[sensor.index] ?? speedByIndex.get(sensor.index) ?? 0),
        ])
      );

      sensorSource.setData({
        type: "FeatureCollection",
        features: sensors.map((sensor) => {
          const speed = speedByIndex.get(sensor.index) ?? 0;
          const baseline = baselineByIndex.get(sensor.index) ?? speed;
          const delta = speed - baseline;

          return {
            type: "Feature",
            geometry: {
              type: "Point",
              coordinates: [sensor.longitude, sensor.latitude],
            },
            properties: {
              sensorId: sensor.sensor_id,
              index: sensor.index,
              speed,
              delta,
              signal: currentHorizon === 15 ? speed : delta,
              selected: sensor.index === currentSelected,
            },
          };
        }),
      });

      const edgeRecords = graph.features.map((feature) => {
        const fromIndex = feature.properties.fromIndex;
        const toIndex = feature.properties.toIndex;

        const fromSpeed = speedByIndex.get(fromIndex) ?? 0;
        const toSpeed = speedByIndex.get(toIndex) ?? 0;

        const fromBaseline = baselineByIndex.get(fromIndex) ?? fromSpeed;
        const toBaseline = baselineByIndex.get(toIndex) ?? toSpeed;

        const speed = (fromSpeed + toSpeed) / 2;
        const baseline = (fromBaseline + toBaseline) / 2;
        const delta = speed - baseline;

        return {
          ...feature,
          properties: {
            ...feature.properties,
            speed,
            delta,
            signal: currentHorizon === 15 ? speed : delta,
          },
          forecastMagnitude: Math.abs(delta),
        };
      });

      // 15m shows the complete monitored network.
      // Longer horizons progressively focus on the road segments
      // with the strongest predicted change from the 15m baseline.
      const keepRatio =
        currentHorizon === 15 ? 1.0 :
        currentHorizon === 30 ? 0.75 :
        currentHorizon === 45 ? 0.55 :
        0.40;

      const sortedEdges = [...edgeRecords].sort(
        (a, b) => b.forecastMagnitude - a.forecastMagnitude
      );

      const visibleCount = Math.max(
        1,
        Math.round(sortedEdges.length * keepRatio)
      );

      const visibleEdges = sortedEdges
        .slice(0, visibleCount)
        .map(({ forecastMagnitude, ...feature }) => feature);

      graphSource.setData({
        type: "FeatureCollection",
        features: visibleEdges,
      });
    };

  
  let particleAnimationFrame = null;
  let particleEdges = [];

  const startTrafficParticles = () => {
    const currentMap = mapRef.current;
    const graph = graphDataRef.current;
    const sensors = sensorDataRef.current;
    const source = currentMap?.getSource("urbanflow-flow-particles");

    if (!currentMap || !graph || !sensors || !source) {
      return;
    }

    const sensorLookup = new Map(
      sensors.map((sensor) => [sensor.index, sensor])
    );

    particleEdges = graph.features
      .filter((feature) => {
        const from = sensorLookup.get(feature.properties.fromIndex);
        const to = sensorLookup.get(feature.properties.toIndex);
        return Boolean(from && to);
      })
      .filter((_, index) => index % 2 === 0)
      .map((feature, index) => {
        const from = sensorLookup.get(feature.properties.fromIndex);
        const to = sensorLookup.get(feature.properties.toIndex);

        return {
          from: [from.longitude, from.latitude],
          to: [to.longitude, to.latitude],
          fromIndex: from.index,
          phase: (index % 17) / 17
        };
      });

    const animateParticles = (time) => {
      const map = mapRef.current;
      const particleSource =
        map?.getSource("urbanflow-flow-particles");

      if (!particleSource || !particleEdges.length) {
        particleAnimationFrame =
          requestAnimationFrame(animateParticles);
        return;
      }

      const values = forecastRef.current.values || [];

      const features = particleEdges.map((edge) => {
        const speed = Number(
          values[edge.fromIndex] ?? 45
        );

        const speedFactor = Math.max(
          0.45,
          Math.min(1.35, speed / 45)
        );

        const t =
          (edge.phase + (time / 3000) * speedFactor) % 1;

        const longitude =
          edge.from[0] +
          (edge.to[0] - edge.from[0]) * t;

        const latitude =
          edge.from[1] +
          (edge.to[1] - edge.from[1]) * t;

        const tone =
          speed < 30
            ? "red"
            : speed < 45
              ? "amber"
              : speed >= 55
                ? "green"
                : "cyan";

        return {
          type: "Feature",
          geometry: {
            type: "Point",
            coordinates: [longitude, latitude]
          },
          properties: {
            speed,
            tone
          }
        };
      });

      particleSource.setData({
        type: "FeatureCollection",
        features
      });

      particleAnimationFrame =
        requestAnimationFrame(animateParticles);
    };

    particleAnimationFrame =
      requestAnimationFrame(animateParticles);
  };

  applyForecastRef.current = applyForecast;

    let animationFrame = null;

  const animateUrbanFlow = (time) => {
    const currentMap = mapRef.current;

    if (
      !currentMap ||
      !currentMap.getLayer("urbanflow-graph-pulse")
    ) {
      animationFrame = requestAnimationFrame(animateUrbanFlow);
      return;
    }

    // Safe MapLibre animation:
    // animate opacity + width only. No line-gradient / line-progress.
    const phase = (time % 2600) / 2600;
    const pulse = (Math.sin(phase * Math.PI * 2) + 1) / 2;

    const opacity = 0.08 + pulse * 0.24;
    const width = 1.2 + pulse * 1.8;

    currentMap.setPaintProperty(
      "urbanflow-graph-pulse",
      "line-opacity",
      opacity
    );

    currentMap.setPaintProperty(
      "urbanflow-graph-pulse",
      "line-width",
      width
    );

    animationFrame = requestAnimationFrame(animateUrbanFlow);
  };

  animationFrame = requestAnimationFrame(animateUrbanFlow);

  map.addControl(
      new maplibregl.NavigationControl({
        showCompass: false,
        showZoom: true,
        visualizePitch: false,
      }),
      "top-right"
    );

    map.on("load", async () => {
      try {
        const [sensorResponse, graphResponse] = await Promise.all([
          fetch("/sensor-locations.json"),
          fetch("/urbanflow-visual-graph.json"),
        ]);

        const sensors = await sensorResponse.json();
        const graph = await graphResponse.json();

        sensorDataRef.current = sensors;
        graphDataRef.current = graph;

      setTimeout(() => {
        startTrafficParticles();
      }, 500);

        map.addSource("urbanflow-graph", {
          type: "geojson",
        lineMetrics: true,
          data: {
            type: "FeatureCollection",
            features: graph.features.map((feature) => ({
              ...feature,
              properties: {
                ...feature.properties,
                speed: 0,
                delta: 0,
                signal: 0,
              },
            })),
          },
        });

        map.addLayer({
          id: "urbanflow-graph-base",
          type: "line",
          source: "urbanflow-graph",
          paint: {
            "line-color": "#8aa3ad",
            "line-width": 1.1,
            "line-opacity": 0.12,
          },
        });

        map.addLayer({
          id: "urbanflow-graph-flow",
          type: "line",
          source: "urbanflow-graph",
          paint: {
            "line-color": forecastColor,
            "line-width": forecastWidth,
            "line-opacity": forecastOpacity,
          },
        });
      
      // Animated traffic propagation signal.
      // Visual-only overlay; forecast density and colors remain unchanged.
      map.addLayer({
        id: "urbanflow-graph-pulse",
        type: "line",
        source: "urbanflow-graph",
        paint: {
          "line-color": "rgba(255,255,255,0)",
          "line-width": [
            "interpolate",
            ["linear"],
            ["zoom"],
            8, 1.0,
            12, 1.8,
            16, 2.8
          ],
          "line-opacity": 0
        }
      });
      
      // Visible moving traffic particles.
      map.addSource("urbanflow-flow-particles", {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: []
        }
      });

      map.addLayer({
        id: "urbanflow-flow-particles",
        type: "circle",
        source: "urbanflow-flow-particles",
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["zoom"],
            8, 4.0,
            12, 5.5,
            16, 7.0
          ],
          "circle-color": [
            "match",
            ["get", "tone"],
            "red", "#ff4d5a",
            "amber", "#ffb347",
            "green", "#45e6a8",
            "cyan", "#6fe7ff",
            "#6fe7ff"
          ],
          "circle-opacity": 1,
          "circle-blur": 0.05,
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": 1.2,
          "circle-stroke-opacity": 0.9
        }
      });



        map.addSource("urbanflow-sensors", {
          type: "geojson",
          data: {
            type: "FeatureCollection",
            features: sensors.map((sensor) => ({
              type: "Feature",
              geometry: {
                type: "Point",
                coordinates: [sensor.longitude, sensor.latitude],
              },
              properties: {
                sensorId: sensor.sensor_id,
                index: sensor.index,
                speed: 0,
                delta: 0,
                signal: 0,
                selected: sensor.index === forecastRef.current.selected,
              },
            })),
          },
        });

        map.addLayer({
          id: "urbanflow-sensor-glow",
          type: "circle",
          source: "urbanflow-sensors",
          paint: {
            "circle-radius": [
              "case",
              ["==", ["get", "selected"], true],
              12,
              8,
            ],
            "circle-color": speedColor,
            "circle-opacity": 0.18,
            "circle-blur": 1,
          },
        });

        map.addLayer({
          id: "urbanflow-sensors",
          type: "circle",
          source: "urbanflow-sensors",
          paint: {
            "circle-radius": [
              "case",
              ["==", ["get", "selected"], true],
              6.5,
              4.5,
            ],
            "circle-color": speedColor,
            "circle-stroke-width": [
              "case",
              ["==", ["get", "selected"], true],
              2.5,
              1.5,
            ],
            "circle-stroke-color": "#ffffff",
            "circle-opacity": 0.95,
          },
        });

        map.on("click", "urbanflow-sensors", (event) => {
          const feature = event.features?.[0];
          if (!feature) return;

          const index = Number(feature.properties.index);
          const sensorId = feature.properties.sensorId;
          const speed = Number(feature.properties.speed || 0);

          onSelectRef.current(index);

          new Popup({ closeButton: true, offset: 10 })
            .setLngLat(feature.geometry.coordinates)
            .setHTML(
              `<div style="font-family:Inter,Arial,sans-serif;min-width:120px">
                <strong>${sensorId}</strong>
                <div style="margin-top:4px;color:#52616b">${speed.toFixed(1)} mph predicted</div>
              </div>`
            )
            .addTo(map);
        });

        map.on("mouseenter", "urbanflow-sensors", () => {
          map.getCanvas().style.cursor = "pointer";
        });

        map.on("mouseleave", "urbanflow-sensors", () => {
          map.getCanvas().style.cursor = "";
        });

        // Critical: apply the current forecast AFTER MapLibre sources exist.
        applyForecast();
      } catch (error) {
        console.error("UrbanFlow map data error:", error);
      }
    });

    return () => {
      applyForecastRef.current = null;
    if (animationFrame) {
      cancelAnimationFrame(animationFrame);
      animationFrame = null;
    }
    if (particleAnimationFrame) {
      cancelAnimationFrame(particleAnimationFrame);
      particleAnimationFrame = null;
    }


      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    // Runs whenever the selected horizon, forecast values,
    // baseline, or selected sensor changes.
    applyForecastRef.current?.();
  }, [values, baselineValues, horizon, selected]);

  return <div ref={mapContainer} className="urban-map" />;
}
