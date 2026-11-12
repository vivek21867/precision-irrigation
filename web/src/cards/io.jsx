import { ArrowRight, ChevronRight } from "lucide-react";
import { SectionLabel } from "../components/ui.jsx";
import { fmt } from "../lib/format.js";
import { navigate } from "../lib/router.js";

const digits = (unit) => (unit === "m³/m³" ? 3 : unit === "W/m²" || unit === "%" ? 0 : 1);

function Step({ eyebrow, title, sub, tone = "bg-fill" }) {
  return (
    <div className={`min-w-0 flex-1 rounded-2xl px-4 py-4 ${tone}`}>
      <div className="eyebrow mb-1.5">{eyebrow}</div>
      <div className="num text-[17px] font-semibold leading-tight tracking-title">{title}</div>
      <div className="mt-1 text-[12px] text-ink-3">{sub}</div>
    </div>
  );
}

/** Summary: what goes into the networks and what comes out. */
export function IOSummary({ data, model }) {
  const io = data.model_io;
  const params = io.models[model]?.n_trainable_params;
  return (
    <div className="flex flex-col items-stretch gap-2 sm:flex-row sm:items-center">
      <Step eyebrow="Input" title={`${io.features.length} features`} sub={`× ${io.input_window} hourly steps`} />
      <ChevronRight size={16} className="mx-auto shrink-0 rotate-90 text-ink-4 sm:rotate-0" aria-hidden />
      <Step eyebrow="Model" title={model} sub={params ? `${params.toLocaleString()} parameters` : "trained network"} tone="bg-sand-soft" />
      <ChevronRight size={16} className="mx-auto shrink-0 rotate-90 text-ink-4 sm:rotate-0" aria-hidden />
      <Step eyebrow="Output" title="Soil moisture 20 cm" sub={`m³/m³ · ${io.horizon} h ahead`} tone="bg-terracotta-soft" />
      <ChevronRight size={16} className="mx-auto shrink-0 rotate-90 text-ink-4 sm:rotate-0" aria-hidden />
      <Step eyebrow="Decision" title="Irrigate / Hold" sub={`below ${fmt(data.threshold.value)} → irrigate`} tone="bg-sage-soft" />
    </div>
  );
}

/** Drill-down: every input feature with unit, latest value and training range. */
export function IODetail({ data }) {
  const io = data.model_io;
  return (
    <div className="space-y-6">
      <SectionLabel>Inputs · each one for every hour of the last {io.input_window} h</SectionLabel>
      <div className="-mx-2 overflow-x-auto px-2">
        <table className="num w-full min-w-[620px] text-left text-[14px]">
          <thead>
            <tr className="text-[12px] text-ink-3">
              <th className="pb-3 font-medium">Feature</th>
              <th className="pb-3 font-medium">Unit</th>
              <th className="pb-3 text-right font-medium">Latest</th>
              <th className="pb-3 text-right font-medium">Training range</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {io.features.map((f) => (
              <tr key={f.key}>
                <td className="py-3">
                  <div className="font-medium">{f.label}</div>
                  <div className="text-[12px] text-ink-3">{f.key}</div>
                </td>
                <td className="text-ink-2">{f.unit}</td>
                <td className="text-right font-semibold">{fmt(f.latest, digits(f.unit))}</td>
                <td className="text-right text-ink-2">{fmt(f.train_min, digits(f.unit))} – {fmt(f.train_max, digits(f.unit))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex flex-col gap-4 rounded-2xl bg-fill p-5 sm:flex-row sm:items-center">
        <p className="text-[13px] leading-relaxed text-ink-2">
          <span className="font-semibold text-ink">Output:</span> volumetric water content at 20 cm for the next hour (m³/m³).
          {data.target_mode === "delta" && " The network predicts the 1-hour change, added to the latest 20 cm reading."}
        </p>
        <button
          type="button"
          onClick={() => navigate("/predict")}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-ink px-4 py-2 text-[13px] font-medium text-white transition hover:bg-ink-2"
        >
          Try your own inputs <ArrowRight size={14} />
        </button>
      </div>
    </div>
  );
}
