import { defineConfig } from "astro/config";

// Static pages; every view fetches live data from the GigScore API in the browser.
// The API address comes from PUBLIC_API_URL (set it in Vercel's project settings).
export default defineConfig({
  output: "static",
  server: { port: 4321 },
});
