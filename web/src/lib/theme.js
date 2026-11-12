// Single source for chart colours: warm neutrals + terracotta / sage / sand. No blues.
export const palette = {
  ink: "#1C1B1A",
  ink2: "#57534E",
  ink3: "#8F8A83",
  ink4: "#C4BFB8",
  fill: "#F2F1EE",
  terracotta: "#C4673F",
  coral: "#E07A5F",
  sage: "#6F9273",
  sand: "#C9A26B",
  clay: "#9A7B66",
};

export const modelColor = {
  Persistence: palette.ink4,
  LSTM: palette.sand,
  BiLSTM: palette.sage,
  Transformer: palette.terracotta,
};

// Soil depths: light sand (shallow) -> deep clay.
export const depthColor = {
  SOIL_MOISTURE_5: "#E3C9A0",
  SOIL_MOISTURE_10: "#C9A26B",
  SOIL_MOISTURE_20: "#C4673F",
  SOIL_MOISTURE_50: "#8C9F7C",
  SOIL_MOISTURE_100: "#6B5A4E",
};

export const spring = { type: "spring", stiffness: 380, damping: 34, mass: 0.9 };
export const softSpring = { type: "spring", stiffness: 260, damping: 30 };

export const axis = {
  tick: { fill: palette.ink3, fontSize: 11 },
  axisLine: false,
  tickLine: false,
};
