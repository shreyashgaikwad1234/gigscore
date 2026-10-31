// Small motion helpers shared by both layouts: numbers that count up when they first appear.
// Any .kpi .value, .stat .v or [data-count] element is animated once, keeping its formatting
// ("₹3.0 Cr", "+19,703", "57.9%"). Skipped when the visitor prefers reduced motion.
const reduce = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
const SELECTOR = ".kpi .value, .stat .v, [data-count]";

export function countUp(el: HTMLElement, ms = 900) {
  if (reduce || el.dataset.counted) return;
  const text = el.textContent || "";
  const m = text.match(/-?[\d,]*\.?\d+/);
  if (!m || (m.index || 0) > 2) return; // only values that start with the number ("+19,703", "₹3.0 Cr"), not names like "lgbm-fair-2026.10"
  el.dataset.counted = "1";
  const raw = m[0], target = Number(raw.replace(/,/g, ""));
  if (!Number.isFinite(target) || target === 0) return;
  const decimals = (raw.split(".")[1] || "").length, grouped = raw.includes(",");
  const [pre, post] = [text.slice(0, m.index), text.slice((m.index || 0) + raw.length)];
  const fmt = (v: number) => grouped ? v.toLocaleString("en-IN", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : v.toFixed(decimals);
  const t0 = performance.now();
  const step = (t: number) => {
    const k = Math.min(1, Math.max(0, (t - t0) / ms)), e = 1 - Math.pow(1 - k, 3); // rAF time can precede t0
    el.textContent = pre + fmt(target * e) + post;
    if (k < 1) requestAnimationFrame(step); else el.textContent = text;
  };
  requestAnimationFrame(step);
}

export function autoCountUp() {
  const io = new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) { countUp(e.target as HTMLElement); io.unobserve(e.target); }
  }), { threshold: 0.4 });
  // values that re-render on every slider move sit inside [data-nocount] and are left alone
  const scan = () => document.querySelectorAll<HTMLElement>(SELECTOR).forEach((n) => {
    if (!n.dataset.counted && !n.dataset.watched && !n.closest("[data-nocount]")) { n.dataset.watched = "1"; io.observe(n); }
  });
  scan();
  new MutationObserver(() => { // values rendered later (after an API call) are re-armed when their text changes
    document.querySelectorAll<HTMLElement>(SELECTOR).forEach((n) => { if (n.dataset.watched && !n.dataset.counted && !n.isConnected) delete n.dataset.watched; });
    scan();
  }).observe(document.body, { childList: true, subtree: true });
}
