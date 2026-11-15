/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        studio: {
          bg: "#0d0f12",
          panel: "#15181e",
          border: "#232730",
          card: "#1a1e26",
          accent: "#3b82f6",
          accentHover: "#2563eb",
          text: "#f3f4f6",
          muted: "#9ca3af",
          subtle: "#4b5563"
        }
      }
    },
  },
  plugins: [],
}

