import { AnimatePresence, motion } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, Check, Droplet, LoaderCircle, RotateCcw, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Pill, Segmented } from "../components/ui.jsx";
import { fmt, fullLabel, hourLabel, signed } from "../lib/format.js";
import { modelColor, spring } from "../lib/theme.js";
import { reveal } from "./Overview.jsx";

const MODES = ["Latest hour", "Steady", "Full 24 h"];
const MODE_NOTE = {
  "Latest hour": (m) => `Your values replace the most recent hour. The previous ${m.input_window - 1} hours are the observed record.`,
  Steady: (m) => `The same values are used for all ${m.input_window} hours, so the model sees no recent trend.`,
  "Full 24 h": (m) => `Edit every hour of the ${m.input_window}-hour window. Paste rows from a spreadsheet into any cell.`,
};
const WEATHER = ["T_HR_AVG", "P_CALC", "SOLARAD", "SUR_TEMP", "RH_HR_AVG"];
const digits = (unit) => (unit === "m³/m³" ? 3 : unit === "W/m²" || unit === "%" ? 0 : 1);
const str = (v, unit) => String(Number(v.toFixed(digits(unit) + 1)));

/** Per-value status: "error" outside physical bounds, "warn" outside training range. */
function check(raw, f) {
  if (raw === "" || raw == null) return { state: "error", num: NaN };
  const num = Number(raw);
  if (!Number.isFinite(num)) return { state: "error", num };
  if (num < f.bounds[0] || num > f.bounds[1]) return { state: "error", num };
  if (num < f.train_min || num > f.train_max) return { state: "warn", num };
  return { state: "ok", num };
}

function Field({ f, value, onChange }) {
  const { state } = check(value, f);
  const ring = { ok: "focus-within:ring-terracotta/40", warn: "ring-1 ring-sand", error: "ring-1 ring-coral" }[state];
  return (
    <label className="block min-w-0">
      <span className="mb-1.5 block truncate text-[13px] font-medium text-ink-2">{f.label}</span>
      <span className={`flex items-center rounded-xl bg-white/80 px-3.5 shadow-[inset_0_0_0_1px_rgba(28,27,26,0.07)] transition focus-within:ring-2 ${ring}`}>
        <input
          inputMode="decimal"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="num h-11 w-full min-w-0 bg-transparent text-[16px] font-semibold tracking-title text-ink outline-none"
          aria-invalid={state === "error"}
        />
        <span className="ml-2 shrink-0 text-[12px] text-ink-3">{f.unit}</span>
      </span>
      <span className={`num mt-1.5 block text-[11px] ${state === "error" ? "text-terracotta" : state === "warn" ? "text-clay" : "text-ink-3"}`}>
        {state === "error"
          ? `Allowed ${f.bounds[0]} – ${f.bounds[1]}`
          : `Trained on ${fmt(f.train_min, digits(f.unit))} – ${fmt(f.train_max, digits(f.unit))}`}
      </span>
    </label>
  );
}

function Group({ title, features, values, setValue }) {
  return (
    <section className="material rounded-card p-6 shadow-card sm:p-7">
      <h2 className="mb-6 text-[15px] font-semibold tracking-title">{title}</h2>
      <div className="grid grid-cols-1 gap-x-5 gap-y-6 sm:grid-cols-2 xl:grid-cols-3">
        {features.map((f) => <Field key={f.key} f={f} value={values[f.key]} onChange={(v) => setValue(f.key, v)} />)}
      </div>
    </section>
  );
}

function WindowGrid({ meta, grid, setGrid }) {
  const feats = meta.features;
  const onPaste = (r0, c0) => (e) => {
    const text = e.clipboardData.getData("text");
    if (!/[\t,\n]/.test(text)) return; // single value: normal paste
    e.preventDefault();
    const rows = text.trim().split(/\r?\n/).map((l) => l.split(/\t|,/).map((s) => s.trim()));
    setGrid((g) => g.map((row, r) => row.map((v, c) => {
      const src = rows[r - r0]?.[c - c0];
      return r >= r0 && c >= c0 && src !== undefined && src !== "" ? src : v;
    })));
  };
  return (
    <section className="material rounded-card p-4 shadow-card sm:p-6">
      <div className="-mx-1 overflow-x-auto px-1">
        <table className="num w-full min-w-[980px] border-separate border-spacing-0 text-[13px]">
          <thead>
            <tr>
              <th className="sticky left-0 bg-white/90 pb-3 pr-3 text-left text-[11px] font-medium text-ink-3">Hour</th>
              {feats.map((f) => (
                <th key={f.key} className="px-1 pb-3 text-left text-[11px] font-medium leading-tight text-ink-3">
                  {f.label}<span className="block font-normal text-ink-4">{f.unit}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.map((row, r) => (
              <tr key={r}>
                <td className="sticky left-0 bg-white/90 py-0.5 pr-3 text-[12px] text-ink-3">
                  {r === grid.length - 1 ? "Now" : `−${grid.length - 1 - r} h`}
                </td>
                {row.map((v, c) => {
                  const st = check(v, feats[c]).state;
                  return (
                    <td key={c} className="px-0.5 py-0.5">
                      <input
                        inputMode="decimal"
                        value={v}
                        onPaste={onPaste(r, c)}
                        onChange={(e) => setGrid((g) => g.map((rr, i) => (i === r ? rr.map((x, j) => (j === c ? e.target.value : x)) : rr)))}
                        className={`h-8 w-full min-w-[72px] rounded-lg px-2 text-[13px] outline-none transition focus:bg-white focus:ring-2 focus:ring-terracotta/40 ${
                          st === "error" ? "bg-coral-soft text-terracotta" : st === "warn" ? "bg-sand-soft" : "bg-fill/70"}`}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Result({ result, model, loading, invalid, meta }) {
  if (!result) {
    return (
      <div className="material flex min-h-[320px] items-center justify-center rounded-card p-7 text-[14px] text-ink-3 shadow-card">
        {invalid ? "Fix the highlighted inputs" : <LoaderCircle className="animate-spin" size={20} />}
      </div>
    );
  }
  const p = result.predictions[model];
  const irrigate = p.irrigate === "YES";
  const Arrow = p.change >= 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <div className="material relative rounded-card p-7 shadow-card">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-[15px] font-semibold tracking-title">Predicted output</h2>
        {loading && <LoaderCircle className="animate-spin text-ink-3" size={16} />}
      </div>
      <div className={`transition-opacity ${invalid ? "opacity-40" : ""}`}>
        <div className="eyebrow mb-2">Soil moisture 20 cm · next hour</div>
        <motion.div key={p.value} initial={{ opacity: 0.4, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={spring}
                    className="display text-[56px]">
          {fmt(p.value)}<span className="ml-1 text-[18px] font-medium tracking-normal text-ink-3">m³/m³</span>
        </motion.div>
        <div className="mt-4 flex flex-wrap gap-2">
          <Pill tone={p.change >= 0 ? "sage" : "terracotta"}><Arrow size={13} strokeWidth={2.5} />{signed(p.change)}</Pill>
          <Pill>From {fmt(result.last_observed)}</Pill>
        </div>

        <div className={`mt-7 flex items-center gap-4 rounded-2xl p-4 ${irrigate ? "bg-terracotta-soft" : "bg-sage-soft"}`}>
          <span className={`flex h-11 w-11 items-center justify-center rounded-[14px] bg-white/70 ${irrigate ? "text-terracotta" : "text-sage"}`}>
            {irrigate ? <Droplet size={20} /> : <Check size={20} />}
          </span>
          <div>
            <div className="text-[19px] font-semibold tracking-title">{irrigate ? "Irrigate" : "Hold"}</div>
            <div className="num text-[12px] text-ink-2">Threshold {fmt(result.threshold.threshold)} · margin {signed(p.margin)}</div>
          </div>
        </div>

        <ul className="mt-6 divide-y divide-line">
          {Object.entries(result.predictions).map(([name, q]) => (
            <li key={name} className={`num flex items-center gap-3 py-3 text-[14px] ${name === model ? "font-semibold" : ""}`}>
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: modelColor[name] }} />
              {name}
              <span className="ml-auto text-[12px] font-normal text-ink-3">{q.irrigate === "YES" ? "Irrigate" : "Hold"}</span>
              <span className="w-14 text-right">{fmt(q.value)}</span>
            </li>
          ))}
        </ul>
      </div>

      <AnimatePresence>
        {result.warnings.length > 0 && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden">
            <div className="mt-5 space-y-2 rounded-2xl bg-sand-soft p-4">
              {result.warnings.map((w) => (
                <p key={w} className="flex gap-2 text-[12px] leading-snug text-clay">
                  <TriangleAlert size={14} className="mt-px shrink-0" />{w}
                </p>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <p className="mt-5 text-[11px] leading-relaxed text-ink-3">
        {meta.target_mode === "delta" ? "Change is added to the 20 cm value of the final input hour. " : ""}
        Irrigate/Hold is a threshold rule, not a water amount.
      </p>
    </div>
  );
}

export default function Predict({ model, setModel, models }) {
  const [meta, setMeta] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [mode, setMode] = useState(MODES[0]);
  const [values, setValues] = useState({});
  const [grid, setGrid] = useState([]);
  const [result, setResult] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [loading, setLoading] = useState(false);
  const reqId = useRef(0);

  const resetFrom = (m) => {
    const last = m.latest_window.rows.at(-1);
    setValues(Object.fromEntries(m.features.map((f, j) => [f.key, str(last[j], f.unit)])));
    setGrid(m.latest_window.rows.map((row) => row.map((v, j) => str(v, m.features[j].unit))));
  };

  useEffect(() => {
    fetch("/api/meta")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((m) => { setMeta(m); resetFrom(m); })
      .catch((e) => setApiError(e));
  }, []);

  // Assemble the 24 x 11 window the networks expect.
  const inputWindow = useMemo(() => {
    if (!meta || !grid.length) return null;
    const row = meta.features.map((f) => values[f.key]);
    if (mode === "Latest hour") return [...meta.latest_window.rows.slice(0, -1).map((r) => r.map(String)), row];
    if (mode === "Steady") return Array.from({ length: meta.input_window }, () => row);
    return grid;
  }, [meta, values, grid, mode]);

  const invalid = useMemo(
    () => !inputWindow || inputWindow.some((r) => r.some((v, j) => check(v, meta.features[j]).state === "error")),
    [inputWindow, meta],
  );

  // Live prediction (debounced) whenever the inputs are valid.
  useEffect(() => {
    if (!inputWindow || invalid) return undefined;
    const id = ++reqId.current;
    const t = setTimeout(() => {
      setLoading(true);
      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ window: inputWindow.map((r) => r.map(Number)) }),
      })
        .then(async (r) => ({ ok: r.ok, body: await r.json() }))
        .then(({ ok, body }) => {
          if (id !== reqId.current) return;
          if (ok) { setResult(body); setServerError(null); } else setServerError(body.error);
        })
        .catch((e) => id === reqId.current && setServerError(e.message))
        .finally(() => id === reqId.current && setLoading(false));
    }, 300);
    return () => clearTimeout(t);
  }, [inputWindow, invalid]);

  if (apiError) {
    return (
      <div className="material mx-auto max-w-xl rounded-card p-8 shadow-card">
        <h1 className="text-[22px] font-semibold tracking-title">Prediction server not running</h1>
        <p className="mt-3 text-[14px] leading-relaxed text-ink-2">
          Manual predictions run the trained PyTorch models. From the project root start
          {" "}<code className="rounded-md bg-fill px-1.5 py-0.5 text-[13px]">python -m src.api</code>, then reload this page.
        </p>
      </div>
    );
  }
  if (!meta || !inputWindow) return <div className="h-[60vh]" />;

  const byKey = Object.fromEntries(meta.features.map((f) => [f.key, f]));
  const setValue = (k, v) => setValues((s) => ({ ...s, [k]: v }));
  const lastTime = meta.latest_window.times.at(-1);

  return (
    <>
      <motion.header {...reveal(0)} className="mb-10 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="eyebrow mb-3">{meta.features.length} inputs × {meta.input_window} h → 1 output</p>
          <h1 className="text-[40px] font-semibold leading-[1.05] tracking-display sm:text-[52px]">Manual Prediction</h1>
          <p className="mt-3 max-w-xl text-[15px] text-ink-2">Enter the conditions the model expects and get the next-hour soil moisture forecast.</p>
        </div>
        <Segmented id="model-p" options={models} value={model} onChange={setModel} />
      </motion.header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.7fr)_minmax(320px,1fr)] lg:gap-7">
        <motion.div {...reveal(1)} className="min-w-0 space-y-6">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <Segmented id="mode" options={MODES} value={mode} onChange={setMode} />
            <button type="button" onClick={() => resetFrom(meta)}
                    className="inline-flex items-center gap-1.5 self-start rounded-full px-3 py-1.5 text-[13px] font-medium text-ink-2 transition hover:bg-fill hover:text-ink sm:self-auto">
              <RotateCcw size={14} /> Reset to observed
            </button>
          </div>
          <p className="text-[13px] leading-relaxed text-ink-3">
            {MODE_NOTE[mode](meta)}{" "}
            {mode === "Latest hour" && <span className="num">Record ends {fullLabel(lastTime)} UTC; your hour stands in for {hourLabel(lastTime)}.</span>}
          </p>

          <motion.div key={mode === "Full 24 h" ? "grid" : "form"} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2 }} className="space-y-6">
              {mode === "Full 24 h" ? (
                <WindowGrid meta={meta} grid={grid} setGrid={setGrid} />
              ) : (
                <>
                  <Group title="Weather" features={WEATHER.map((k) => byKey[k])} values={values} setValue={setValue} />
                  <Group title="Soil" features={meta.features.filter((f) => !WEATHER.includes(f.key))} values={values} setValue={setValue} />
                </>
              )}
          </motion.div>
        </motion.div>

        <motion.aside {...reveal(2)} className="lg:sticky lg:top-20 lg:self-start">
          <Result result={result} model={model} loading={loading} invalid={invalid} meta={meta} />
          {serverError && <p className="mt-3 px-2 text-[12px] text-terracotta">{serverError}</p>}
        </motion.aside>
      </div>
    </>
  );
}
