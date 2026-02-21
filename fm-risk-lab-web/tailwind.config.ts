import type { Config } from "tailwindcss";

export default {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ["var(--font-mono)", "JetBrains Mono", "Fira Code", "monospace"],
      },
      colors: {
        "accent-cyan":  "#22d3ee",
        "accent-amber": "#f59e0b",
        "accent-red":   "#ef4444",
        "card-bg":      "#0f172a",
        "card-border":  "#1e3a5f",
      },
    },
  },
  plugins: [],
} satisfies Config;
