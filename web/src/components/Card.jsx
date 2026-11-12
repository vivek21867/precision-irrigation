import { AnimatePresence, motion } from "framer-motion";
import { GripVertical, Plus } from "lucide-react";
import { softSpring, spring } from "../lib/theme.js";

/**
 * Glass card. Static cards only summarise; cards with `detail` expand in place
 * (spring layout animation) to full width and reveal drill-down content.
 */
export default function Card({
  title, icon: Icon, accent = "text-ink-3", expanded = false, onToggle, detail, children,
  drag = {}, dragging = false, layoutEnabled = true, className = "",
}) {
  const expandable = typeof detail === "function";
  const toggle = () => expandable && onToggle?.();

  return (
    <motion.article
      layout={layoutEnabled}
      transition={spring}
      animate={{ scale: dragging ? 1.015 : 1 }}
      onPointerDown={drag.onPointerDown}
      onClick={toggle}
      onKeyDown={(e) => {
        if (!expandable) return;
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
        if (e.key === "Escape" && expanded) toggle();
      }}
      role={expandable ? "button" : undefined}
      tabIndex={expandable ? 0 : undefined}
      aria-expanded={expandable ? expanded : undefined}
      style={{ borderRadius: 22 }}
      className={[
        "group material relative h-full overflow-hidden p-6 sm:p-7 select-none",
        "transition-shadow duration-300",
        dragging ? "shadow-lift z-10 cursor-grabbing" : "shadow-card",
        expandable ? "cursor-pointer hover:shadow-lift" : "cursor-grab",
        className,
      ].join(" ")}
    >
      <motion.header layout="position" transition={spring} className="mb-5 flex items-center gap-2.5">
        {Icon && <Icon size={17} strokeWidth={2} className={accent} aria-hidden />}
        <h2 className="text-[15px] font-semibold tracking-title text-ink">{title}</h2>
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            aria-label={`Move ${title}`}
            {...drag.handle}
            onClick={(e) => e.stopPropagation()}
            className="flex h-7 w-7 cursor-grab items-center justify-center rounded-full text-ink-4 opacity-0 transition-opacity duration-200 hover:bg-fill group-hover:opacity-100 focus-visible:opacity-100"
          >
            <GripVertical size={15} />
          </button>
          {expandable && (
            <motion.span
              animate={{ rotate: expanded ? 45 : 0 }}
              transition={spring}
              className="flex h-7 w-7 items-center justify-center rounded-full bg-fill text-ink-2"
              aria-hidden
            >
              <Plus size={15} strokeWidth={2.25} />
            </motion.span>
          )}
        </div>
      </motion.header>

      <motion.div layout="position" transition={spring}>{children}</motion.div>

      <AnimatePresence initial={false}>
        {expanded && expandable && (
          <motion.div
            key="detail"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0, transition: { ...softSpring, delay: 0.08 } }}
            exit={{ opacity: 0, y: 8, transition: { duration: 0.15 } }}
            onClick={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
            className="mt-8 cursor-auto select-text border-t border-line pt-8"
          >
            {detail()}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.article>
  );
}
