// Shared browser helpers: API client and formatting. Every page imports these.
export const API: string = (import.meta.env.PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export async function api(path: string, opts: RequestInit = {}): Promise<any> {
  const res = await fetch(API + path, {
    ...opts,
    headers: { "Content-Type": "application/json", ...((opts.headers as Record<string, string>) || {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = (body as any).detail;
    throw new Error(Array.isArray(d) ? d.map((e: any) => `${e.loc?.slice(-1)[0]}: ${e.msg}`).join("; ") : d || res.statusText);
  }
  return body;
}

export const $ = <T extends Element = HTMLElement>(s: string, el: ParentNode = document) => el.querySelector(s) as T;
export const $$ = <T extends Element = HTMLElement>(s: string, el: ParentNode = document) => [...el.querySelectorAll(s)] as T[];

export const esc = (v: unknown) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
export const pct = (v: number | null | undefined, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined) => (v == null ? "–" : Number(v).toLocaleString("en-IN"));
export const inr = (v: number | null | undefined) => (v == null ? "–" : `₹${Number(v).toLocaleString("en-IN")}`);
export const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Static copies of the API's read-only responses, written by `python -m src.snapshot` into public/snapshot.
// The free API sleeps when idle and takes about a minute to wake; pages show these instead of waiting.
export const snapshot = (name: string): Promise<any> =>
  fetch(`/snapshot/${name}.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`snapshot ${name} missing`))));

/** True once any page request has fallen back to a snapshot because the API was slow to answer. */
export let usedSnapshot = false;

/** The live response if it arrives within `wait` ms, otherwise the snapshot (the live call keeps waking the API). */
export async function liveOr(path: string, snap: string, opts: RequestInit = {}, wait = 2500): Promise<any> {
  const live = api(path, opts);
  live.catch(() => {});
  const late = Symbol();
  try {
    const r = await Promise.race([live, new Promise((res) => setTimeout(res, wait, late))]);
    if (r !== late) return r;
  } catch { /* API unreachable: use the snapshot */ }
  usedSnapshot = true;
  return snapshot(snap);
}

/** Health pill for both layouts. If /health has not answered within 3 s the API is waking from sleep: say so with a
 *  running count and, where every screen needs the API (the workspace), a notice that stays until it answers. */
export function trackHealth(pill: HTMLElement, live: (h: any) => [string, string], notice = false) {
  const t0 = Date.now();
  let toast: HTMLElement | null = null, tick = 0;
  const draw = () => {
    const s = `${Math.round((Date.now() - t0) / 1000)}s`;
    pill.textContent = `● Waking up · ${s}`;
    if (toast) toast.querySelector("b")!.textContent = s;
  };
  const slow = setTimeout(() => {
    pill.className = "pill-status warn";
    if (notice) {
      toast = document.createElement("div");
      toast.className = "toast";
      toast.setAttribute("role", "status");
      toast.innerHTML = `<span>Starting the demonstration API — free hosting pauses it when idle. It takes about a minute, and this page fills in by itself. <b></b></span>`;
      document.body.appendChild(toast);
    }
    draw();
    tick = window.setInterval(draw, 1000);
  }, 3000);
  const done = () => { clearTimeout(slow); clearInterval(tick); toast?.remove(); };
  api("/health")
    .then((h) => { done(); const [text, cls] = live(h); pill.textContent = text; pill.className = "pill-status " + cls; })
    .catch(() => { done(); pill.textContent = "● API offline"; pill.className = "pill-status err"; });
}

let modelCache: any = null;
export async function getModel(force = false) {
  // force: settings pages need the current live policy, not the published copy
  if (!modelCache || force) modelCache = force ? await api("/model") : await liveOr("/model", "model");
  return modelCache;
}

let analyticsCache: any = null;
export async function getAnalytics() {
  // validation aggregates change only on retraining, and the snapshot is regenerated with them
  if (!analyticsCache) analyticsCache = await snapshot("analytics").catch(() => api("/model/analytics"));
  return analyticsCache;
}

export const BAND_COPY: Record<string, string> = {
  approve: "Approve",
  refer: "Refer to credit officer",
  decline: "Decline",
};

export const ACTION_COPY: Record<string, string> = {
  none: "No action required — continue weekly monitoring.",
  reminder_call: "Courtesy reminder call before the next due date.",
  emi_date_shift: "Offer to align the EMI date with the borrower's income cycle.",
  restructuring_offer: "Contact the borrower to discuss restructuring before an EMI is missed.",
};

export function showError(el: HTMLElement, e: unknown) {
  el.textContent = e instanceof Error ? e.message : String(e);
  el.hidden = false;
}
