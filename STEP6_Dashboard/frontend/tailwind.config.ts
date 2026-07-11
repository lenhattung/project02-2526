import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17202a",
        line: "#d9e2ec",
        brand: "#1f7a8c",
        accent: "#bf6f13",
        success: "#2f855a",
        danger: "#c2410c"
      },
      boxShadow: {
        panel: "0 1px 2px rgba(16, 24, 40, 0.06)"
      }
    },
  },
  plugins: [],
} satisfies Config;
