import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api is served by the local Python prediction server (python -m src.api).
const api = { "/api": { target: "http://127.0.0.1:8000", changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, proxy: api },
  preview: { proxy: api },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          charts: ["recharts"],
          motion: ["framer-motion", "@dnd-kit/core", "@dnd-kit/sortable"],
        },
      },
    },
  },
});
