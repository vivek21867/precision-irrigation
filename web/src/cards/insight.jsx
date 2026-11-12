import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTip, SectionLabel, Stat } from "../components/ui.jsx";
import { dateLabel, FEATURE, fmt } from "../lib/format.js";
import { axis, palette } from "../lib/theme.js";

const label = (f) => (f.startsWith("SOIL_MOISTURE") ? `Soil ${FEATURE[f].label}` : FEATURE[f].label);

function HBars({ rows, color, height, digits = 4, highlight }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={rows} layout="vertical" margin={{ left: 0, right: 12 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="name" width={96} {...axis} tick={{ ...axis.tick, fontSize: 12, fill: palette.ink2 }} />
          <Bar dataKey="v" radius={8} barSize={10} background={{ fill: palette.fill, radius: 8 }} isAnimationActive={false}>
            {rows.map((r, i) => <Cell key={r.name} fill={i === 0 && highlight ? color : palette.ink4} />)}
          </Bar>
          <Tooltip cursor={false} content={<ChartTip rows={(p) => p.map((x) => [x.payload.name, fmt(x.value, digits), color])} />} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Summary: top drivers by permutation importance (target history excluded — it dominates by construction). */
export function DriversSummary({ data, model }) {
  const rows = data.permutation_importance
    .filter((r) => r.model === model && r.feature !== data.target)
    .sort((a, b) => b.rmse_increase_mean - a.rmse_increase_mean)
    .slice(0, 5)
    .map((r) => ({ name: label(r.feature), v: r.rmse_increase_mean }));
  return (
    <div>
      <Stat value={rows[0]?.name ?? "—"} size="md" label="Strongest driver after 20 cm history" />
      <div className="mt-6">
        <HBars rows={rows} color={palette.terracotta} height={150} highlight />
      </div>
    </div>
  );
}

/** Drill-down: full permutation ranking and mean |SHAP| for the selected model. */
export function DriversDetail({ data, model }) {
  const perm = data.permutation_importance
    .filter((r) => r.model === model)
    .sort((a, b) => b.rmse_increase_mean - a.rmse_increase_mean)
    .map((r) => ({ name: label(r.feature), v: r.rmse_increase_mean }));
  const shap = Object.entries(data.shap_by_feature[model] ?? {})
    .map(([f, v]) => ({ name: label(f), v }))
    .sort((a, b) => b.v - a.v);
  return (
    <div className="grid gap-10 lg:grid-cols-2">
      <div>
        <SectionLabel>Permutation · Δ test RMSE</SectionLabel>
        <HBars rows={perm} color={palette.terracotta} height={330} digits={5} highlight />
      </div>
      <div>
        <SectionLabel>SHAP · mean |attribution| on Δ</SectionLabel>
        <HBars rows={shap} color={palette.sage} height={330} digits={5} highlight />
      </div>
      <p className="text-[12px] leading-relaxed text-ink-3 lg:col-span-2">
        20 cm history ranks first by construction: the forecast adds the predicted change to its last value.
        Attributions describe model behaviour, not causality.
      </p>
    </div>
  );
}

/** Static: how much of the NOAA record is usable. */
export function DatasetSummary({ data }) {
  const ds = data.dataset;
  const years = Object.entries(ds.rows_by_year);
  const max = Math.max(...years.map(([, n]) => n));
  const splits = [["Train", ds.splits.train, palette.sand], ["Validation", ds.splits.val, palette.sage], ["Test", ds.splits.test, palette.terracotta]];
  const total = splits.reduce((s, [, v]) => s + v.windows, 0);
  return (
    <div className="grid gap-8 sm:grid-cols-[auto_minmax(0,1fr)] sm:gap-12">
      <div>
        <Stat value={ds.clean_rows.toLocaleString()} size="lg" label={`of ${ds.raw_rows.toLocaleString()} hours complete`} />
        <div className="mt-6 flex h-16 items-end gap-1.5">
          {years.map(([y, n]) => (
            <div key={y} className="flex flex-col items-center gap-1.5">
              <div className="w-6 rounded-md bg-sand" style={{ height: `${Math.max(4, (n / max) * 44)}px` }} title={`${y}: ${n.toLocaleString()} h`} />
              <span className="num text-[10px] text-ink-3">{y.slice(2)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="flex flex-col justify-between gap-6">
        <div>
          <div className="flex h-2.5 overflow-hidden rounded-full">
            {splits.map(([k, v, c]) => <div key={k} style={{ width: `${(v.windows / total) * 100}%`, background: c }} />)}
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3">
            {splits.map(([k, v, c]) => (
              <div key={k} className="min-w-0">
                <div className="flex items-center gap-1.5 text-[12px] font-medium text-ink-3">
                  <span className="h-2 w-2 rounded-full" style={{ background: c }} />{k}
                </div>
                <div className="num mt-1 text-[17px] font-semibold tracking-title">{v.windows.toLocaleString()}</div>
                <div className="num truncate text-[11px] text-ink-3">{dateLabel(v.end)}</div>
              </div>
            ))}
          </div>
        </div>
        <p className="text-[12px] text-ink-3">Chronological split · {ds.segments} contiguous segments · sensor gaps not filled</p>
      </div>
    </div>
  );
}
