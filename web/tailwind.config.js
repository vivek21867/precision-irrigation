/** Warm-neutral, light-only palette. No blue hues anywhere. */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    colors: {
      transparent: "transparent",
      current: "currentColor",
      white: "#FFFFFF",
      canvas: "#FAFAF9",
      fill: { DEFAULT: "#F2F1EE", strong: "#E8E6E1" },
      ink: { DEFAULT: "#1C1B1A", 2: "#57534E", 3: "#8F8A83", 4: "#C4BFB8" },
      line: "rgba(28,27,26,0.06)",
      terracotta: { DEFAULT: "#C4673F", soft: "#F6E5DC" },
      coral: { DEFAULT: "#E07A5F", soft: "#FBE9E3" },
      sage: { DEFAULT: "#6F9273", soft: "#E6EFE5" },
      sand: { DEFAULT: "#C9A26B", soft: "#F5ECDD" },
      clay: "#9A7B66",
    },
    extend: {
      fontFamily: {
        sans: ['"SF Pro Display"', '"SF Pro Text"', "-apple-system", "BlinkMacSystemFont",
               '"Inter Variable"', "Inter", '"Segoe UI"', "system-ui", "sans-serif"],
      },
      borderRadius: { card: "22px" },
      boxShadow: {
        card: "0 1px 1px rgba(60,40,20,0.03), 0 8px 24px -6px rgba(60,40,20,0.08), 0 24px 48px -12px rgba(60,40,20,0.06)",
        lift: "0 2px 4px rgba(60,40,20,0.04), 0 24px 48px -8px rgba(60,40,20,0.16), 0 40px 80px -20px rgba(60,40,20,0.14)",
      },
      letterSpacing: { display: "-0.035em", title: "-0.022em", label: "0.02em" },
    },
  },
  plugins: [],
};
