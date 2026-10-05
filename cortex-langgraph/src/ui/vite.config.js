import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        // Backend address. Override when port 8000 is taken, e.g.
        //   API_TARGET=http://localhost:8001 npm run dev
        target: process.env.API_TARGET || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
