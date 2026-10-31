// Workspace context shared by every workspace page: the branch being viewed and the (demo) signed-in user.
// Kept in localStorage; this is a demo stand-in for real logins.
const get = (k: string) => { try { return localStorage.getItem(k) || ""; } catch { return ""; } };
const set = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* no storage */ } };

export const getBranch = () => get("gigscore-branch");
export const setBranch = (v: string) => set("gigscore-branch", v);
export const getOfficer = () => get("gigscore-officer");
export const setOfficer = (v: string) => set("gigscore-officer", v);

/** "3 h 20 min", "2 days" … since an ISO timestamp. */
export function ago(iso: string) {
  const m = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 60000));
  if (m < 60) return `${m} min`;
  if (m < 24 * 60) return `${Math.floor(m / 60)} h ${m % 60} min`;
  const d = Math.floor(m / 1440);
  return `${d} day${d > 1 ? "s" : ""}`;
}
export const hoursSince = (iso: string) => (Date.now() - Date.parse(iso)) / 3600e3;

export const STATUS = (c: any) =>
  c.disbursed ? ["Disbursed", "approve"] : c.final ? (c.final.status === "approve" ? ["Approved", "approve"] : ["Declined", "decline"])
    : ["Waiting for decision", "refer"];
