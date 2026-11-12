import { Area, AreaChart, Bar, BarChart, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTip, Legend, SectionLabel } from "../components/ui.jsx";
import { dayLabel, FEATURE, fmt, fullLabel } from "../lib/format.js";
import { axis, depthColor, modelColor, palette } from "../lib/theme.js";

const DEPTHS = ["SOIL_MOISTURE_5", "SOIL_MOISTURE_10", "SOIL_MOISTURE_20", "SOIL_MOISTURE_50", "SOIL_MOISTURE_100"];
// Round the y-domain to 0.02 m³/m³ so ticks land on even values.
const niceDomain = ([lo, hi]) => [Math.floor(lo * 50) / 50, Math.ceil(hi * 50) / 50];
const dayTicks = (rows) => rows.filter((r) => r.t.endsWith("T12:00")).map((r) => r.t);

/** 7-day soil moisture at the target depth, with threshold and forecast. */
export function MoistureSummary({ data, model }) {
  const tgt = data.target;
  const rows = data.history.map((r) => ({ t: r.t, v: r[tgt] }));
  rows.push({ t: data.latest.forecast_time, f: data.forecasts[model] });
  rows[rows.length - 2].f = rows[rows.length - 2].v;

  return (
    <div>
      <div className="h-[230px]">
        <ResponsiveContainer>
          <AreaChart data={rows} margin={{ top: 10, right: 6, bottom: 0, left: -12 }}>
            <defs>
              <linearGradient id="sm-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={palette.terracotta} stopOpacity={0.22} />
                <stop offset="100%" stopColor={palette.terracotta} stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="t" ticks={dayTicks(rows)} tickFormatter={dayLabel} {...axis} dy={6} />
            <YAxis domain={niceDomain} tickFormatter={(v) => v.toFixed(2)} tickCount={4} allowDecimals {...axis} width={48} />
            <ReferenceLine y={data.threshold.value} stroke={palette.ink4} strokeDasharray="2 4"
                           label={{ value: "threshold", position: "insideBottomLeft", fill: palette.ink3, fontSize: 11 }} />
            <Area type="monotone" dataKey="v" stroke={palette.terracotta} strokeWidth={2.25} fill="url(#sm-fill)"
                  activeDot={{ r: 4, fill: palette.terracotta, stroke: "#fff", strokeWidth: 2 }} isAnimationActive={false} />
            <Area type="monotone" dataKey="f" stroke={modelColor[model]} strokeDasharray="3 4" strokeWidth={2} fill="none"
                  dot={(p) => (p.index === rows.length - 1
                    ? <circle key="d" cx={p.cx} cy={p.cy} r={4.5} fill="#fff" stroke={modelColor[model]} strokeWidth={2.5} />
                    : <g key={p.index} />)} isAnimationActive={false} />
            <Tooltip cursor={{ stroke: palette.ink4 }}
                     content={<ChartTip title={(l) => fullLabel(l)}
                                        rows={(p) => p.filter((x) => x.value != null).slice(0, 1)
                                          .map((x) => [x.dataKey === "f" ? `${model} forecast` : "Observed", fmt(x.value), palette.terracotta])} />} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** Drill-down: full soil profile (all five depths). */
export function MoistureDetail({ data }) {
  const latest = data.latest.values;
  const bars = DEPTHS.map((f) => ({ f, label: FEATURE[f].label, v: latest[f] }));
  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <div>
        <SectionLabel right={<Legend items={DEPTHS.map((f) => [FEATURE[f].label, depthColor[f]])} />}>Soil profile · 7 days</SectionLabel>
        <div className="h-[260px]">
          <ResponsiveContainer>
            <LineChart data={data.history} margin={{ top: 8, right: 6, left: -12 }}>
              <XAxis dataKey="t" ticks={dayTicks(data.history)} tickFormatter={dayLabel} {...axis} dy={6} />
              <YAxis tickFormatter={(v) => v.toFixed(2)} tickCount={5} {...axis} width={48} domain={["auto", "auto"]} />
              {DEPTHS.map((f) => (
                <Line key={f} type="monotone" dataKey={f} stroke={depthColor[f]} strokeWidth={f === data.target ? 2.5 : 1.75}
                      dot={false} isAnimationActive={false} />
              ))}
              <Tooltip cursor={{ stroke: palette.ink4 }}
                       content={<ChartTip title={fullLabel} rows={(p) => p.map((x) => [FEATURE[x.dataKey].label, fmt(x.value), x.color])} />} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div>
        <SectionLabel>Now · m³/m³</SectionLabel>
        <div className="h-[260px]">
          <ResponsiveContainer>
            <BarChart data={bars} layout="vertical" margin={{ right: 40 }}>
              <XAxis type="number" hide domain={[0, "dataMax"]} />
              <YAxis type="category" dataKey="label" width={56} {...axis} tick={{ ...axis.tick, fontSize: 12, fill: palette.ink2 }} />
              <Bar dataKey="v" radius={10} barSize={18} background={{ fill: palette.fill, radius: 10 }} isAnimationActive={false}
                   label={{ position: "right", formatter: (v) => fmt(v), fill: palette.ink, fontSize: 12, fontWeight: 600 }}>
                {bars.map((b) => <Cell key={b.f} fill={depthColor[b.f]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
