import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    // The name "static" is load-bearing: the repo's root .gitignore has a
    // bare `dist/` entry that matches ANY directory named `dist` at any
    // depth (git.gitignore semantics for a bare pattern with no leading
    // slash). Vite's default outDir is `dist`, which would build fine
    // locally but silently never ship in a wheel. Building into `../static`
    // instead keeps the output git-tracked (RESEARCH.md Pitfall #1).
    outDir: "../static",
    emptyOutDir: true,
  },
  server: {
    port: 5174,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
});
