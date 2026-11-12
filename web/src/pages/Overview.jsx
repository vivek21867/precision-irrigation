import { motion } from "framer-motion";
import {
  Activity, BarChart3, CloudRain, Cpu, Database, Droplet, Gauge, Layers, LayoutGrid, Sparkles, Thermometer,
} from "lucide-react";
import { useMemo, useState } from "react";
import Card from "../components/Card.jsx";
import SortableGrid from "../components/SortableGrid.jsx";
import { Segmented } from "../components/ui.jsx";
import { DecisionSummary } from "../cards/decision.jsx";
import { ConditionsSummary, RainSummary } from "../cards/environment.jsx";
import { ForecastDetail, ForecastSummary } from "../cards/forecast.jsx";
import { DatasetSummary, DriversDetail, DriversSummary } from "../cards/insight.jsx";
import { IODetail, IOSummary } from "../cards/io.jsx";
import { IrrigationDetail, IrrigationSummary } from "../cards/irrigation.jsx";
import { MoistureDetail, MoistureSummary } from "../cards/moisture.jsx";
import { PerformanceDetail, PerformanceSummary } from "../cards/performance.jsx";
import { fullLabel } from "../lib/format.js";
import { softSpring } from "../lib/theme.js";
import { usePersistentState } from "../lib/useDashboard.js";

// Card registry. `Detail` present => card expands on click; absent => static summary.
const CARDS = [
  { id: "forecast", title: "Forecast", icon: Sparkles, accent: "text-terracotta", span: 2, Summary: ForecastSummary, Detail: ForecastDetail },
  { id: "decision", title: "Irrigation", icon: Droplet, accent: "text-terracotta", span: 1, Summary: DecisionSummary },
  { id: "io", title: "Inputs & Output", icon: Cpu, accent: "text-clay", span: 3, Summary: IOSummary, Detail: IODetail },
  { id: "moisture", title: "Soil Moisture · 20 cm", icon: Layers, accent: "text-clay", span: 2, Summary: MoistureSummary, Detail: MoistureDetail },
  { id: "conditions", title: "Conditions", icon: Thermometer, accent: "text-sand", span: 1, Summary: ConditionsSummary },
  { id: "performance", title: "Accuracy", icon: BarChart3, accent: "text-terracotta", span: 1, Summary: PerformanceSummary, Detail: PerformanceDetail },
  { id: "decisions", title: "Decision Quality", icon: Gauge, accent: "text-sage", span: 1, Summary: IrrigationSummary, Detail: IrrigationDetail },
  { id: "drivers", title: "Drivers", icon: Activity, accent: "text-clay", span: 1, Summary: DriversSummary, Detail: DriversDetail },
  { id: "rain", title: "Rainfall", icon: CloudRain, accent: "text-sage", span: 1, Summary: RainSummary },
  { id: "dataset", title: "Dataset", icon: Database, accent: "text-sand", span: 2, Summary: DatasetSummary },
];
const DEFAULT_ORDER = CARDS.map((c) => c.id);
export const reveal = (i) => ({ initial: { opacity: 0, y: 16 }, animate: { opacity: 1, y: 0, transition: { ...softSpring, delay: 0.04 * i } } });

export default function Overview({ data, model, setModel, models }) {
  const [order, setOrder] = usePersistentState("pi.layout.v1", DEFAULT_ORDER);
  // Deep link: /#performance opens that card if it has drill-down detail.
  const [expanded, setExpanded] = useState(() => {
    const id = window.location.hash.slice(1);
    return CARDS.some((c) => c.id === id && c.Detail) ? id : null;
  });

  // Keep stored layouts valid when cards are added or removed.
  const safeOrder = useMemo(() => {
    const kept = order.filter((id) => DEFAULT_ORDER.includes(id));
    for (const id of DEFAULT_ORDER) if (!kept.includes(id)) kept.splice(DEFAULT_ORDER.indexOf(id), 0, id);
    return kept;
  }, [order]);

  const items = CARDS.map((c) => ({
    id: c.id,
    span: c.span,
    render: (dnd) => (
      <Card
        {...dnd}
        title={c.title}
        icon={c.icon}
        accent={c.accent}
        expanded={expanded === c.id}
        onToggle={() => setExpanded((e) => (e === c.id ? null : c.id))}
        detail={c.Detail ? () => <c.Detail data={data} model={model} /> : undefined}
      >
        <c.Summary data={data} model={model} />
      </Card>
    ),
  }));

  return (
    <>
      <motion.header {...reveal(0)} className="mb-10 flex flex-col gap-6 sm:mb-14 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="eyebrow mb-3">{data.station.name.replaceAll("_", " ")} · USCRN</p>
          <h1 className="text-[40px] font-semibold leading-[1.05] tracking-display sm:text-[52px]">Soil Moisture</h1>
          <p className="num mt-3 text-[15px] text-ink-2">Latest complete hour · {fullLabel(data.latest.time)} UTC</p>
        </div>
        <div className="flex items-center gap-3">
          <Segmented id="model" options={models} value={model} onChange={setModel} />
          <button
            type="button"
            onClick={() => { setOrder(DEFAULT_ORDER); setExpanded(null); }}
            aria-label="Reset layout"
            title="Reset layout"
            className="material flex h-9 w-9 items-center justify-center rounded-full text-ink-2 shadow-card transition hover:text-ink"
          >
            <LayoutGrid size={16} />
          </button>
        </div>
      </motion.header>

      <motion.div {...reveal(1)}>
        <SortableGrid items={items} order={safeOrder} onOrderChange={setOrder} expandedId={expanded} />
      </motion.div>

      <motion.footer {...reveal(2)} className="mx-auto mt-16 max-w-3xl text-center text-[12px] leading-relaxed text-ink-3">
        <p>
          Irrigation is a threshold proxy on forecast 20 cm moisture — no irrigation labels exist and no water quantity is predicted.
          Threshold {data.threshold.value.toFixed(3)} m³/m³ ({data.threshold.source}).
        </p>
        <p className="mt-2">
          Data: <a href={data.station.url} className="underline decoration-ink-4 underline-offset-2 hover:text-ink-2" target="_blank" rel="noreferrer">{data.station.source}</a>
          {" "}· historical observations, not live · exported {data.generated_at.replace("T", " ")}
        </p>
      </motion.footer>
    </>
  );
}
