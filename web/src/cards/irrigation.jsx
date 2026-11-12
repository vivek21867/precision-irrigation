import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartTip, Legend, SectionLabel, Stat } from "../components/ui.jsx";
import { fmt, pct } from "../lib/format.js";
import { axis, modelColor, palette } from "../lib/theme.js";

/** Summary: decision quality of the threshold proxy for the selected model. */
export function IrrigationSummary({ data, model }) {
  const m = data.irrigation[model];
  return (
    <div>
      <Stat value={pct(m.recall_irrigate)} size="lg" label="Irrigate recall · test" />
      <div className="mt-7 grid grid-cols-3 gap-4">
        {[["F1", pct(m.f1_irrigate)], ["Precision", pct(m.precision_irrigate)], ["Hours below", pct(m.positive_rate_actual, 0)]].map(([k, v]) => (
          <div key={k}>
            <div className="num text-[17px] font-semibold tracking-title">{v}</div>
            <div className="mt-1 text-[12px] text-ink-3">{k}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Confusion({ cm }) {
  // rows = actual [NO, YES]; cols = predicted [NO, YES]
  const cells = [
    ["Hold · correct", cm[0][0], "bg-sage-soft text-sage"],
    ["Irrigate · false alarm", cm[0][1], "bg-fill text-ink-2"],
    ["Hold · missed", cm[1][0], "bg-coral-soft text-terracotta"],
    ["Irrigate · correct", cm[1][1], "bg-terracotta-soft text-terracotta"],
  ];
  return (
    <div className="grid grid-cols-2 gap-3">
      {cells.map(([label, n, tone]) => (
        <div key={label} className={`rounded-2xl p-4 ${tone}`}>
          <div className="num text-[24px] font-semibold tracking-title">{n.toLocaleString()}</div>
          <div className="mt-1 text-[12px] font-medium opacity-80">{label}</div>
        </div>
      ))}
    </div>
  );
}

/** Drill-down: confusion matrix and recall across thresholds. */
export function IrrigationDetail({ data, model }) {
  const models = Object.keys(data.irrigation);
  const byPct = new Map();
  for (const r of data.threshold_sweep) {
    const row = byPct.get(r.threshold) ?? { thr: r.threshold, pctl: r.train_percentile };
    row[r.model] = r.recall_irrigate;
    byPct.set(r.threshold, row);
  }
  const sweep = [...byPct.values()].sort((a, b) => a.thr - b.thr);

  return (
    <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)]">
      <div>
        <SectionLabel>{model} · actual vs. decided</SectionLabel>
        <Confusion cm={data.irrigation[model].confusion_matrix} />
        <p className="mt-4 text-[12px] leading-relaxed text-ink-3">
          Labels come from the same threshold applied to observed moisture — a proxy, not real irrigation records.
        </p>
      </div>
      <div>
        <SectionLabel right={<Legend items={models.map((m) => [m, modelColor[m], m === "Persistence"])} />}>
          Recall by threshold
        </SectionLabel>
        <div className="h-[240px]">
          <ResponsiveContainer>
            <LineChart data={sweep} margin={{ top: 8, right: 8, left: -12 }}>
              <XAxis dataKey="thr" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => v.toFixed(2)} {...axis} dy={6} />
              <YAxis domain={["auto", 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tickCount={4} {...axis} width={48} />
              <ReferenceLine x={data.threshold.value} stroke={palette.ink4} strokeDasharray="2 4" />
              {models.map((m) => (
                <Line key={m} type="monotone" dataKey={m} stroke={modelColor[m]} strokeWidth={m === model ? 2.5 : 1.5}
                      strokeDasharray={m === "Persistence" ? "3 4" : undefined} dot={false} isAnimationActive={false} />
              ))}
              <Tooltip cursor={{ stroke: palette.ink4 }}
                       content={<ChartTip title={(l, p) => `Threshold ${fmt(l)} · P${p[0]?.payload.pctl}`}
                                          rows={(p) => p.map((x) => [x.dataKey, pct(x.value), x.stroke])} />} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
