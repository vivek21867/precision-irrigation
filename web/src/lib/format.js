export const FEATURE = {
  T_HR_AVG: { label: "Air temp", unit: "°C", digits: 1 },
  P_CALC: { label: "Rain", unit: "mm", digits: 1 },
  SOLARAD: { label: "Solar", unit: "W/m²", digits: 0 },
  SUR_TEMP: { label: "Surface", unit: "°C", digits: 1 },
  RH_HR_AVG: { label: "Humidity", unit: "%", digits: 0 },
  SOIL_TEMP_20: { label: "Soil temp", unit: "°C", digits: 1 },
  SOIL_MOISTURE_5: { label: "5 cm", unit: "m³/m³", digits: 3 },
  SOIL_MOISTURE_10: { label: "10 cm", unit: "m³/m³", digits: 3 },
  SOIL_MOISTURE_20: { label: "20 cm", unit: "m³/m³", digits: 3 },
  SOIL_MOISTURE_50: { label: "50 cm", unit: "m³/m³", digits: 3 },
  SOIL_MOISTURE_100: { label: "100 cm", unit: "m³/m³", digits: 3 },
};

export const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
export const signed = (v, d = 3) => {
  if (v == null) return "—";
  const r = Number(v.toFixed(d));
  return r === 0 ? (0).toFixed(d) : `${r > 0 ? "+" : "−"}${Math.abs(r).toFixed(d)}`;
};
export const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// Timestamps are UTC strings "YYYY-MM-DDTHH:MM"; parse without local-time shifts.
const parts = (s) => {
  const [d, t] = s.split("T");
  const [y, m, day] = d.split("-").map(Number);
  return { y, m, day, hh: t.slice(0, 5) };
};
export const dayLabel = (s) => { const p = parts(s); return `${MONTHS[p.m - 1]} ${p.day}`; };
export const fullLabel = (s) => { const p = parts(s); return `${MONTHS[p.m - 1]} ${p.day}, ${p.y} · ${p.hh}`; };
export const hourLabel = (s) => parts(s).hh;
export const dateLabel = (s) => { const p = parts(s.replace(" ", "T")); return `${MONTHS[p.m - 1]} ${p.day}, ${p.y}`; };
