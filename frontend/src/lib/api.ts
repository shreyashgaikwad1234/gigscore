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

let modelCache: any = null;
export async function getModel(force = false) {
  if (!modelCache || force) modelCache = await api("/model");
  return modelCache;
}

let analyticsCache: any = null;
export async function getAnalytics() {
  if (!analyticsCache) analyticsCache = await api("/model/analytics");
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
