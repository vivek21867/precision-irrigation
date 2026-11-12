import { Bar, BarChart, Cell, LabelList, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTip, Legend, SectionLabel, Stat } from "../components/ui.jsx";
import { dayLabel, fmt, fullLabel } from "../lib/format.js";
import { axis, modelColor, palette } from "../lib/theme.js";

const nets = (d) => d.metrics.filter((m) => m.model !== "Persistence");

/** Summary: RMSE skill vs persistence on the held-out test period. */
export function PerformanceSummary({ data, model }) {
  const sel = data.metrics.find((m) => m.model === model);
  const rows = nets(data).map((m) => ({ name: m.model, skill: m.RMSE_skill_vs_persistence * 100 }));
  return (
    <div>
      <Stat value={`${sel.RMSE_skill_vs_persistence >= 0 ? "+" : ""}${(sel.RMSE_skill_vs_persistence * 100).toFixed(1)}`}
            unit="%" size="lg" label={`${model} · RMSE skill vs. persistence`} />
      <div className="mt-6 h-[120px]">
        <ResponsiveContainer>
          <BarChart data={rows} layout="vertical" margin={{ left: 0, right: 44 }}>
            <XAxis type="number" hide domain={[Math.min(0, ...rows.map((r) => r.skill)), "dataMax"]} />
            <YAxis type="category" dataKey="name" width={84} {...axis} tick={{ ...axis.tick, fontSize: 12, fill: palette.ink2 }} />
            <Bar dataKey="skill" radius={8} barSize={12} background={{ fill: palette.fill, radius: 8 }} isAnimationActive={false}>
              {rows.map((r) => <Cell key={r.name} fill={r.name === model ? modelColor[r.name] : palette.ink4} />)}
              <LabelList dataKey="skill" position="right" formatter={(v) => `${v.toFixed(1)}%`}
                         style={{ fill: palette.ink, fontSize: 12, fontWeight: 600 }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/** Drill-down: metric table, actual vs predicted, training curves. */
export function PerformanceDetail({ data, model }) {
  const tp = data.test_predictions;
  const ticks = tp.filter((r) => r.target_time.endsWith("T12:00")).filter((_, i) => i % 2 === 0).map((r) => r.target_time);
  const hist = data.training_history[model] ?? [];
  const best = Math.min(...data.metrics.map((m) => m.RMSE));

  return (
    <div className="space-y-12">
      <div className="-mx-2 overflow-x-auto px-2">
        <table className="num w-full min-w-[520px] text-left text-[14px]">
          <thead>
            <tr className="text-[12px] text-ink-3">
              {["Model", "MAE", "RMSE", "R²", "Skill"].map((h, i) => (
                <th key={h} className={`pb-3 font-medium ${i ? "text-right" : ""}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {data.metrics.map((m) => (
              <tr key={m.model} className={m.model === model ? "font-semibold" : ""}>
                <td className="py-3">
                  <span className="mr-2.5 inline-block h-2 w-2 rounded-full" style={{ background: modelColor[m.model] }} />
                  {m.model}
                </td>
                <td className="text-right">{fmt(m.MAE, 5)}</td>
                <td className="text-right">
                  {fmt(m.RMSE, 5)}
                  {m.RMSE === best && <span className="ml-1.5 text-terracotta">●</span>}
                </td>
                <td className="text-right">{fmt(m.R2, 4)}</td>
                <td className="text-right">{m.model === "Persistence" ? "—" : `${(m.RMSE_skill_vs_persistence * 100).toFixed(1)}%`}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-[12px] text-ink-3">
          Held-out test · {data.dataset.splits.test.windows.toLocaleString()} hours · m³/m³
        </p>
      </div>

      <div>
        <SectionLabel right={<Legend items={[["Observed", palette.ink], [model, modelColor[model]], ["Persistence", palette.ink4, true]]} />}>
          Test period · last 14 days
        </SectionLabel>
        <div className="h-[260px]">
          <ResponsiveContainer>
            <LineChart data={tp} margin={{ top: 8, right: 6, left: -12 }}>
              <XAxis dataKey="target_time" ticks={ticks} tickFormatter={dayLabel} {...axis} dy={6} />
              <YAxis domain={["auto", "auto"]} tickFormatter={(v) => v.toFixed(2)} tickCount={4} {...axis} width={48} />
              <Line type="monotone" dataKey="Persistence" stroke={palette.ink4} strokeWidth={1.5} strokeDasharray="3 4" dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="actual" stroke={palette.ink} strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey={model} stroke={modelColor[model]} strokeWidth={2} dot={false} isAnimationActive={false} />
              <Tooltip cursor={{ stroke: palette.ink4 }}
                       content={<ChartTip title={fullLabel}
                                          rows={(p) => p.map((x) => [x.dataKey === "actual" ? "Observed" : x.dataKey, fmt(x.value), x.stroke])} />} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {hist.length > 0 && (
        <div>
          <SectionLabel right={<Legend items={[["Train", palette.sand], ["Validation", palette.terracotta]]} />}>
            {model} · training loss (MSE, scaled)
          </SectionLabel>
          <div className="h-[200px]">
            <ResponsiveContainer>
              <LineChart data={hist} margin={{ top: 8, right: 6, left: -12 }}>
                <XAxis dataKey="epoch" {...axis} dy={6} />
                <YAxis {...axis} width={48} tickFormatter={(v) => v.toFixed(2)} tickCount={4} domain={["auto", "auto"]} />
                <Line type="monotone" dataKey="train_loss" stroke={palette.sand} strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="val_loss" stroke={palette.terracotta} strokeWidth={2} dot={false} isAnimationActive={false} />
                <Tooltip cursor={{ stroke: palette.ink4 }}
                         content={<ChartTip title={(l) => `Epoch ${l}`}
                                            rows={(p) => p.map((x) => [x.dataKey === "val_loss" ? "Validation" : "Train", fmt(x.value, 4), x.stroke])} />} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}
