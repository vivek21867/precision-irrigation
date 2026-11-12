import { motion } from "framer-motion";
import { Check, Droplet } from "lucide-react";
import { fmt, signed } from "../lib/format.js";
import { spring } from "../lib/theme.js";

/** Static summary: threshold rule applied to the selected model's forecast. */
export function DecisionSummary({ data, model }) {
  const pred = data.forecasts[model];
  const thr = data.threshold.value;
  const irrigate = pred < thr;
  const Icon = irrigate ? Droplet : Check;
  // Gauge spans the 7-day observed range plus forecast and threshold.
  const seen = [...data.history.map((r) => r[data.target]), pred, thr];
  const lo = Math.min(...seen) - 0.02;
  const hi = Math.max(...seen) + 0.02;
  const pos = (v) => `${((v - lo) / (hi - lo)) * 100}%`;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-4">
        <motion.div
          key={String(irrigate)}
          initial={{ scale: 0.6, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={spring}
          className={`flex h-14 w-14 items-center justify-center rounded-[18px] ${irrigate ? "bg-terracotta-soft text-terracotta" : "bg-sage-soft text-sage"}`}
        >
          <Icon size={26} strokeWidth={2} />
        </motion.div>
        <div>
          <div className="display text-[34px]">{irrigate ? "Irrigate" : "Hold"}</div>
          <div className="mt-1 text-[13px] text-ink-3">{irrigate ? "Below threshold" : "Above threshold"}</div>
        </div>
      </div>

      <div className="mt-auto pt-8">
        <div className="relative h-2 rounded-full bg-gradient-to-r from-terracotta-soft via-fill to-sage-soft">
          <span className="absolute top-1/2 h-4 w-[2px] -translate-y-1/2 rounded-full bg-ink-4" style={{ left: pos(thr) }} />
          <motion.span
            className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white shadow-card ${irrigate ? "bg-terracotta" : "bg-sage"}`}
            initial={false}
            animate={{ left: pos(pred) }}
            transition={spring}
          />
        </div>
        <div className="num mt-3 flex justify-between text-[12px] text-ink-3">
          <span>Threshold {fmt(thr)}</span>
          <span>Margin {signed(pred - thr)}</span>
        </div>
      </div>
    </div>
  );
}
