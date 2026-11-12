import { useEffect, useState } from "react";

// Loads the JSON produced by `python -m src.export_dashboard` (real pipeline outputs).
export function useDashboard() {
  const [state, setState] = useState({ data: null, error: null });
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/dashboard.json`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((data) => setState({ data, error: null }))
      .catch((error) => setState({ data: null, error }));
  }, []);
  return state;
}

export function usePersistentState(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
  }, [key, value]);
  return [value, setValue];
}
