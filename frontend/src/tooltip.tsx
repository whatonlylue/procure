import { useEffect, useLayoutEffect, useRef, useState } from "react";

const DELAY_MS = 700;

type Tip = { text: string; rect: DOMRect };
const MARGIN = 8;

/** Long-hover tooltip: any element with a `data-tip` attribute shows it after a pause. */
export function TooltipLayer() {
  const [tip, setTip] = useState<Tip | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const current = useRef<Element | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null);

  // Measure the rendered tooltip, then place it on whichever side has room and
  // clamp it inside the viewport.
  useLayoutEffect(() => {
    const el = box.current;
    if (!tip || !el) {
      setPos(null);
      return;
    }
    const { width, height } = el.getBoundingClientRect();
    const { rect } = tip;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const gap = 8;
    const fitsAbove = rect.top - gap - height >= MARGIN;
    const fitsBelow = rect.bottom + gap + height <= vh - MARGIN;
    const above = fitsAbove || (!fitsBelow && rect.top > vh - rect.bottom);
    let top = above ? rect.top - gap - height : rect.bottom + gap;
    top = Math.max(MARGIN, Math.min(top, vh - MARGIN - height));
    let left = rect.left + rect.width / 2 - width / 2;
    left = Math.max(MARGIN, Math.min(left, vw - MARGIN - width));
    setPos({ left, top });
  }, [tip]);

  useEffect(() => {
    const clear = () => {
      window.clearTimeout(timer.current);
      current.current = null;
      setTip(null);
    };
    const over = (e: MouseEvent) => {
      const el = (e.target as Element | null)?.closest?.("[data-tip]") ?? null;
      if (el === current.current) return;
      clear();
      const text = el?.getAttribute("data-tip");
      if (!el || !text) return;
      current.current = el;
      timer.current = window.setTimeout(() => {
        setTip({ text, rect: el.getBoundingClientRect() });
      }, DELAY_MS);
    };
    document.addEventListener("mouseover", over);
    document.addEventListener("mousedown", clear);
    document.addEventListener("scroll", clear, true);
    return () => {
      clear();
      document.removeEventListener("mouseover", over);
      document.removeEventListener("mousedown", clear);
      document.removeEventListener("scroll", clear, true);
    };
  }, []);

  if (!tip) return null;
  return (
    <div
      ref={box}
      className="tip"
      role="tooltip"
      style={{ left: pos?.left ?? 0, top: pos?.top ?? 0, visibility: pos ? "visible" : "hidden" }}
    >
      {tip.text}
    </div>
  );
}
