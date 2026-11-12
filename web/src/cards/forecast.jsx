import { Area, AreaChart, Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { ChartTip, Pill, SectionLabel, Stat } from "../components/ui.jsx";
import { fmt, fullLabel, hourLabel, signed } from "../lib/format.js";
import { axis, modelColor, palette } from "../lib/theme.js";

const observedOf = (d) => d.latest.values[d.target];

/** Hero summary: next-hour forecast from the selected model. */
export function ForecastSummary({ data, model }) {
  const obs = observedOf(data);
  const pred = data.forecasts[model];
  const delta = pred - obs;
  const Arrow = delta >= 0 ? ArrowUpRight : ArrowDownRight;
  const series = [
    ...data.input_window.map((p) => ({ t: p.t, v: p.v })),
    { t: data.latest.forecast_time, f: pred },
  ];
  series[series.length - 2].f = series[series.length - 2].v; // join observed line to forecast

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)] lg:items-end">
      <div className="space-y-6">
        <Stat value={fmt(pred)} unit="m³/m³" size="xl" label={`Next hour · ${fullLabel(data.latest.forecast_time)} UTC`} />
        <div className="flex flex-wrap items-center gap-2">
          <Pill tone={delta >= 0 ? "sage" : "terracotta"}>
            <Arrow size={13} strokeWidth={2.5} />
            {signed(delta)}
          </Pill>
          <Pill>Now {fmt(obs)}</Pill>
        </div>
      </div>
      <div className="h-[132px]">
        <ResponsiveContainer>
          <AreaChart data={series} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
            <defs>
              <linearGradient id="fc-fill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={palette.terracotta} stopOpacity={0.18} />
                <stop offset="100%" stopColor={palette.terracotta} stopOpacity={0} />
              </linearGradient>
            </defs>
            <YAxis hide domain={["dataMin - 0.004", "dataMax + 0.004"]} />
            <XAxis dataKey="t" hide />
            <ReferenceLine y={data.threshold.value} stroke={palette.ink4} strokeDasharray="2 4" ifOverflow="hidden" />
            <Area type="monotone" dataKey="v" stroke={palette.ink2} strokeWidth={2} fill="url(#fc-fill)" isAnimationActive={false} />
            <Area type="monotone" dataKey="f" stroke={modelColor[model]} strokeWidth={2} strokeDasharray="4 4" fill="none"
                  dot={(p) => (p.index === series.length - 1
                    ? <circle key="dot" cx={p.cx} cy={p.cy} r={5} fill={modelColor[model]} stroke="#fff" strokeWidth={2} />
                    : <g key={p.index} />)}
                  isAnimationActive={false} />
            <Tooltip cursor={{ stroke: palette.ink4, strokeWidth: 1 }}
                     content={<ChartTip title={(l) => `${hourLabel(l)} UTC`}
                                        rows={(p) => p.filter((x) => x.value != null).slice(0, 1)
                                          .map((x) => [x.dataKey === "f" ? model : "Observed", fmt(x.value), x.color])} />} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** Drill-down: every model's forecast for the same window. */
export function ForecastDetail({ data }) {
  const obs = observedOf(data);
  const rows = Object.entries(data.forecasts).map(([name, v]) => ({ name, v, d: v - obs }));
  return (
    <div className="grid gap-10 lg:grid-cols-2">
      <div>
        <SectionLabel>All models · same 24 h window</SectionLabel>
        <ul className="divide-y divide-line">
          {rows.map((r) => (
            <li key={r.name} className="flex items-center gap-3 py-3.5">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: modelColor[r.name] }} />
              <span className="text-[14px] font-medium">{r.name}</span>
              <span className="num ml-auto text-[13px] text-ink-3">{signed(r.d)}</span>
              <span className="num w-16 text-right text-[15px] font-semibold">{fmt(r.v)}</span>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <SectionLabel>Change vs. now · m³/m³</SectionLabel>
        <div className="h-[220px]">
          <ResponsiveContainer>
            <BarChart data={rows} layout="vertical" margin={{ left: 0, right: 24 }}>
              <XAxis type="number" hide domain={["auto", "auto"]} />
              <YAxis type="category" dataKey="name" width={92} {...axis} tick={{ ...axis.tick, fontSize: 12, fill: palette.ink2 }} />
              <ReferenceLine x={0} stroke={palette.ink4} />
              <Bar dataKey="d" radius={8} barSize={14} isAnimationActive={false}>
                {rows.map((r) => <Cell key={r.name} fill={modelColor[r.name]} />)}
              </Bar>
              <Tooltip cursor={{ fill: palette.fill }}
                       content={<ChartTip rows={(p) => p.map((x) => [x.payload.name, signed(x.value, 4), modelColor[x.payload.name]])} />} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-3 text-[12px] text-ink-3">
          {data.target_mode === "delta" ? "Networks predict the 1-hour change, added to the last observation." : "Networks predict the level directly."}
          {" "}Persistence repeats the last value.
        </p>
      </div>
    </div>
  );
}
