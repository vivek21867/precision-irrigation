import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";
import { CloudRain, Droplets, Leaf, Sun, Thermometer, Waves } from "lucide-react";
import { ChartTip, Stat } from "../components/ui.jsx";
import { dayLabel, FEATURE, fmt } from "../lib/format.js";
import { axis, palette } from "../lib/theme.js";

const ICON = { T_HR_AVG: Thermometer, RH_HR_AVG: Droplets, P_CALC: CloudRain, SOLARAD: Sun, SUR_TEMP: Waves, SOIL_TEMP_20: Leaf };

/** Static: latest hourly conditions. */
export function ConditionsSummary({ data }) {
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-7">
      {data.conditions.map((f) => {
        const Icon = ICON[f];
        return (
          <div key={f} className="min-w-0">
            <div className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-ink-3">
              {Icon && <Icon size={14} strokeWidth={2} aria-hidden />}
              {FEATURE[f].label}
            </div>
            <Stat value={fmt(data.latest.values[f], FEATURE[f].digits)} unit={FEATURE[f].unit} size="sm" />
          </div>
        );
      })}
    </div>
  );
}

/** Static: daily rainfall totals over the 7-day history. */
export function RainSummary({ data }) {
  const days = new Map();
  for (const r of data.history) {
    const d = r.t.slice(0, 10);
    days.set(d, (days.get(d) ?? 0) + (r.P_CALC ?? 0));
  }
  const rows = [...days].map(([d, mm]) => ({ t: `${d}T00:00`, mm }));
  const total = rows.reduce((s, r) => s + r.mm, 0);
  return (
    <div>
      <Stat value={fmt(total, 1)} unit="mm" size="lg" label="Last 7 days" />
      <div className="mt-6 h-[120px]">
        <ResponsiveContainer>
          <BarChart data={rows} margin={{ left: 0, right: 0 }}>
            <XAxis dataKey="t" tickFormatter={(t) => dayLabel(t).split(" ")[1]} {...axis} interval={0} />
            <Bar dataKey="mm" fill={palette.sage} radius={[7, 7, 7, 7]} barSize={16}
                 background={{ fill: palette.fill, radius: 7 }} isAnimationActive={false} />
            <Tooltip cursor={false} content={<ChartTip title={dayLabel} rows={(p) => p.map((x) => ["Rain", `${fmt(x.value, 1)} mm`, palette.sage])} />} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
