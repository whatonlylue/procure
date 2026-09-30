export type ThemeName = "ember" | "paper" | "moss" | "lagoon";

export interface ThemeDef {
  id: ThemeName;
  label: string;
  blurb: string;
  /** Page background, used for the theme-color meta tag. */
  bg: string;
}

export const THEMES: ThemeDef[] = [
  { id: "ember", label: "Ember", blurb: "Dark with a signal-amber accent", bg: "#141512" },
  { id: "paper", label: "Paper", blurb: "Light with a bronze-amber accent", bg: "#e8eae4" },
  { id: "moss", label: "Moss", blurb: "Dark with a moss-green accent", bg: "#12150f" },
  { id: "lagoon", label: "Lagoon", blurb: "Dark with a lagoon-teal accent", bg: "#101416" },
];

export const THEME_KEY = "procure-theme";

export const DEFAULT_THEME: ThemeName = "ember";

function isTheme(v: unknown): v is ThemeName {
  return typeof v === "string" && THEMES.some((t) => t.id === v);
}

export function getInitialTheme(): ThemeName {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (isTheme(stored)) return stored;
  } catch {
    // Storage unavailable (private mode, SSR) — fall through to default.
  }
  return DEFAULT_THEME;
}

export function applyTheme(theme: ThemeName): void {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Ignore persistence failures; the theme still applies for this session.
  }
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) {
    const def = THEMES.find((t) => t.id === theme);
    if (def) meta.setAttribute("content", def.bg);
  }
}
