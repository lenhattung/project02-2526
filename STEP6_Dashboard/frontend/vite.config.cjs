const { defineConfig } = require("vite");
const react = require("@vitejs/plugin-react");

module.exports = defineConfig({
  plugins: [react()],
  cacheDir: ".vite-cache",
  build: {
    outDir: "../frontend-dist",
  },
  server: {
    host: "0.0.0.0",
    port: 4173,
    strictPort: true,
  },
  preview: {
    host: "0.0.0.0",
    port: 4174,
    strictPort: true,
  },
});
