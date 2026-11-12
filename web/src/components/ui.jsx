import { motion } from "framer-motion";
import { spring } from "../lib/theme.js";

/** iOS-style segmented control with a sliding thumb. */
export function Segmented({ options, value, onChange, id = "seg", size = "md" }) {
  const pad = size === "sm" ? "px-3 py-1 text-[12px]" : "px-4 py-1.5 text-[13px]";
  return (
    <div
      role="radiogroup"
      className="inline-flex rounded-full bg-fill p-[3px]"
      onClick={(e) => e.stopPropagation()}
      onPointerDown={(e) => e.stopPropagation()}
    >
      {options.map((o) => {
        const active = o === value;
        return (
          <button
            key={o}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o)}
            className={`relative rounded-full font-medium transition-colors ${pad} ${active ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}
          >
            {active && (
              <motion.span
                layoutId={`${id}-thumb`}
                transition={spring}
                className="absolute inset-0 rounded-full bg-white shadow-[0_1px_3px_rgba(60,40,20,0.12),0_1px_1px_rgba(60,40,20,0.04)]"
              />
            )}
            <span className="relative">{o}</span>
          </button>
        );
      })}
    </div>
  );
}

/** Big number with unit and a small label underneath. */
export function Stat({ value, unit, label, size = "md", tone = "text-ink" }) {
  const sz = { xl: "text-[56px] sm:text-[64px]", lg: "text-[34px]", md: "text-[26px]", sm: "text-[22px]" }[size];
  const unitSz = { xl: "text-[20px]", lg: "text-[15px]", md: "text-[13px]", sm: "text-[12px]" }[size];
  return (
    <div className="min-w-0">
      <div className={`display ${sz} ${tone}`}>
        {value}
        {unit && <span className={`ml-1 font-medium tracking-normal text-ink-3 ${unitSz}`}>{unit}</span>}
      </div>
      {label && <div className="mt-2 text-[13px] text-ink-3">{label}</div>}
    </div>
  );
}

export function Pill({ children, tone = "neutral" }) {
  const tones = {
    neutral: "bg-fill text-ink-2",
    terracotta: "bg-terracotta-soft text-terracotta",
    sage: "bg-sage-soft text-sage",
    sand: "bg-sand-soft text-clay",
  };
  return (
    <span className={`num inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-[12px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

/** Frosted tooltip for Recharts. `rows` maps payload -> [label, value, color]. */
export function ChartTip({ active, payload, label, title, rows }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="material rounded-2xl px-3.5 py-2.5 shadow-lift">
      <div className="mb-1 text-[11px] font-medium text-ink-3">{title ? title(label, payload) : label}</div>
      {rows(payload).map(([name, value, color]) => (
        <div key={name} className="num flex items-center gap-2 text-[13px]">
          <span className="h-2 w-2 rounded-full" style={{ background: color }} />
          <span className="text-ink-2">{name}</span>
          <span className="ml-auto pl-4 font-semibold text-ink">{value}</span>
        </div>
      ))}
    </div>
  );
}

export function Legend({ items }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5">
      {items.map(([name, color, dashed]) => (
        <span key={name} className="flex items-center gap-1.5 text-[12px] text-ink-2">
          <span
            className="h-[3px] w-3.5 rounded-full"
            style={dashed ? { backgroundImage: `linear-gradient(90deg, ${color} 50%, transparent 50%)`, backgroundSize: "5px 3px" } : { background: color }}
          />
          {name}
        </span>
      ))}
    </div>
  );
}

export function SectionLabel({ children, right }) {
  return (
    <div className="mb-4 flex items-baseline justify-between gap-4">
      <h3 className="eyebrow">{children}</h3>
      {right}
    </div>
  );
}
