import { Droplet } from "lucide-react";
import { useEffect } from "react";
import { Segmented } from "./components/ui.jsx";
import Overview from "./pages/Overview.jsx";
import Predict from "./pages/Predict.jsx";
import { navigate, usePath } from "./lib/router.js";
import { useDashboard, usePersistentState } from "./lib/useDashboard.js";

const ROUTES = { Overview: "/", Predict: "/predict" };

function Shell({ children, path }) {
  const current = path.startsWith("/predict") ? "Predict" : "Overview";
  return (
    <div className="relative min-h-screen overflow-x-hidden">
      {/* Warm ambient wash behind the glass cards. */}
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute -left-40 -top-40 h-[520px] w-[520px] rounded-full bg-terracotta-soft opacity-70 blur-[120px]" />
        <div className="absolute -right-32 top-1/3 h-[460px] w-[460px] rounded-full bg-sage-soft opacity-80 blur-[120px]" />
        <div className="absolute bottom-0 left-1/3 h-[420px] w-[420px] rounded-full bg-sand-soft opacity-80 blur-[120px]" />
      </div>
      <nav className="material sticky top-0 z-30 shadow-[0_1px_0_rgba(28,27,26,0.05)]">
        <div className="mx-auto flex h-14 max-w-[1320px] items-center justify-between gap-4 px-4 sm:px-8 lg:px-12">
          <button type="button" onClick={() => navigate("/")} className="flex items-center gap-2 text-[15px] font-semibold tracking-title">
            <span className="flex h-7 w-7 items-center justify-center rounded-[9px] bg-terracotta text-white">
              <Droplet size={15} strokeWidth={2.4} />
            </span>
            <span className="hidden sm:inline">Precision Irrigation</span>
          </button>
          <Segmented id="nav" size="sm" options={Object.keys(ROUTES)} value={current} onChange={(k) => navigate(ROUTES[k])} />
        </div>
      </nav>
      <main className="mx-auto max-w-[1320px] px-4 pb-24 pt-10 sm:px-8 sm:pt-14 lg:px-12">{children}</main>
    </div>
  );
}

export default function App() {
  const path = usePath();
  const { data, error } = useDashboard();
  const [model, setModel] = usePersistentState("pi.model.v1", null);

  useEffect(() => {
    if (data && (!(model in data.forecasts) || model === "Persistence")) setModel(data.default_model);
  }, [data, model, setModel]);

  if (error) {
    return (
      <Shell path={path}>
        <div className="material mx-auto max-w-lg rounded-card p-8 shadow-card">
          <h1 className="text-[22px] font-semibold tracking-title">No dashboard data</h1>
          <p className="mt-3 text-[14px] text-ink-2">
            Run <code className="rounded-md bg-fill px-1.5 py-0.5 text-[13px]">python -m src.export_dashboard</code> from the project root, then reload.
          </p>
        </div>
      </Shell>
    );
  }
  if (!data || !(model in data.forecasts)) return <Shell path={path}><div className="h-[60vh]" /></Shell>;

  const props = { data, model, setModel, models: Object.keys(data.forecasts).filter((m) => m !== "Persistence") };
  return (
    <Shell path={path}>
      {path.startsWith("/predict") ? <Predict key="predict" {...props} /> : <Overview key="overview" {...props} />}
    </Shell>
  );
}
