import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development, proxy /api calls to the FastAPI backend on :8000 so the
// SPA and API share an origin. In production the backend serves the built
// assets directly. `base: "./"` makes every asset path relative, so the built
// bundle (including the standalone demo.html) works whether it's served from
// FastAPI at `/` or embedded under any sub-path on a static host.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      // Paths are resolved relative to the Vite project root.
      input: {
        main: "index.html",
        demo: "demo.html",
      },
    },
  },
});
