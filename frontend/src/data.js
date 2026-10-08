export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export const HORIZONS = [15, 30, 45, 60];

export function average(values = []) {
  return values.length ? values.reduce((sum, value) => sum + Number(value || 0), 0) / values.length : 0;
}

export function minValue(values = []) {
  return values.length ? Math.min(...values) : 0;
}

export function maxValue(values = []) {
  return values.length ? Math.max(...values) : 0;
}

export function trafficState(speed) {
  if (speed < 30) return { key: "pressure", label: "Pressure", tone: "red" };
  if (speed < 45) return { key: "moderate", label: "Moderate", tone: "amber" };
  if (speed < 55) return { key: "moving", label: "Moving", tone: "blue" };
  return { key: "free", label: "Free flow", tone: "green" };
}

export function buildSensorRecords(predictions = {}) {
  const values15 = predictions["15_min"] || [];
  const values30 = predictions["30_min"] || [];
  const values45 = predictions["45_min"] || [];
  const values60 = predictions["60_min"] || [];

  return values15.map((speed, index) => ({
    id: `S-${String(index + 1).padStart(3, "0")}`,
    index,
    speed15: Number(speed),
    speed30: Number(values30[index] ?? speed),
    speed45: Number(values45[index] ?? speed),
    speed60: Number(values60[index] ?? speed),
    state: trafficState(Number(speed)),
  }));
}

export function horizonValues(predictions, horizon) {
  return predictions?.[`${horizon}_min`] || [];
}
