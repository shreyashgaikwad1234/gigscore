// Reads a bank / UPI statement exported as CSV into the rows the /watch API expects.
// Handles the usual Indian bank export shapes: account details above the header row, quoted amounts
// with commas ("1,234.00"), day-first dates (03/08/2026, 03-Aug-26), separate Withdrawal / Deposit
// columns or one Amount column with Dr / Cr, and a Narration column (the API infers bills and bounces from it).

export type StatementRow = { date: string; credit: number; debit: number; balance: number; description?: string; category?: string; status?: string };
export type Parsed = { rows: StatementRow[]; columns: Record<string, string>; skipped: number; from: string; to: string };

const SYN: Record<string, string[]> = {
  date: ["txn date", "transaction date", "tran date", "date", "posting date", "value date", "value dt"],
  description: ["narration", "description", "particulars", "transaction details", "transaction remarks", "remarks", "details"],
  debit: ["withdrawal amt", "withdrawal amount", "withdrawals", "withdrawal", "debit amount", "debit amt", "debit", "dr amount", "money out", "paid out"],
  credit: ["deposit amt", "deposit amount", "deposits", "deposit", "credit amount", "credit amt", "credit", "cr amount", "money in", "paid in"],
  balance: ["closing balance", "running balance", "available balance", "balance", "bal"],
  amount: ["transaction amount", "txn amount", "amount"],
  type: ["dr cr", "cr dr", "debit credit", "transaction type", "type"],
  category: ["category"],
  status: ["status"],
};

const norm = (h: string) => h.toLowerCase().replace(/\(.*?\)/g, " ").replace(/inr|rs\.?|₹/g, " ").replace(/[^a-z0-9]+/g, " ").trim();

function splitCSV(text: string): string[][] {
  const first = text.split(/\r?\n/).slice(0, 30).join("\n");
  const delim = [",", ";", "\t"].sort((a, b) => first.split(b).length - first.split(a).length)[0];
  const out: string[][] = [];
  let row: string[] = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') q = false; else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === delim) { row.push(cell); cell = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell); out.push(row); row = []; cell = "";
    } else cell += ch;
  }
  if (cell || row.length) { row.push(cell); out.push(row); }
  return out.map((r) => r.map((c) => c.trim())).filter((r) => r.some((c) => c));
}

function findColumns(header: string[]) {
  const h = header.map(norm), col: Record<string, number> = {};
  for (const [key, names] of Object.entries(SYN))
    for (const n of names) {
      const i = h.findIndex((x, j) => !Object.values(col).includes(j) && (x === n || x.startsWith(n + " ")));
      if (i >= 0) { col[key] = i; break; }
    }
  return col;
}

const MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];
export function parseDate(s: string): string | null {
  const t = s.trim();
  let m = t.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  let y: number, mo: number, d: number;
  if (m) [y, mo, d] = [+m[1], +m[2], +m[3]];
  else if ((m = t.match(/^(\d{1,2})[\/\-. ](\d{1,2})[\/\-. ](\d{2,4})/))) [d, mo, y] = [+m[1], +m[2], +m[3]]; // day first (India)
  else if ((m = t.match(/^(\d{1,2})[\/\-. ]([A-Za-z]{3})[A-Za-z]*[\/\-., ]+(\d{2,4})/))) [d, mo, y] = [+m[1], MONTHS.indexOf(m[2].toLowerCase()) + 1, +m[3]];
  else return null;
  if (y < 100) y += 2000;
  if (!(mo >= 1 && mo <= 12 && d >= 1 && d <= 31)) return null;
  return `${y}-${String(mo).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}

export function parseAmount(s: string | undefined): number {
  if (!s) return 0;
  let t = s.replace(/[₹,\s]|inr|rs\.?/gi, "");
  const neg = /^\(.*\)$/.test(t) || /dr$/i.test(t) || t.startsWith("-");
  t = t.replace(/[()]|cr$|dr$/gi, "").replace(/^[-+]/, "");
  const v = Number(t);
  return Number.isFinite(v) ? (neg ? -v : v) : NaN;
}

export function parseStatement(text: string): Parsed {
  const table = splitCSV(text.replace(/^﻿/, ""));
  const hi = table.slice(0, 40).findIndex((r) => {
    const c = findColumns(r);
    return c.date !== undefined && (c.balance !== undefined) && (c.credit !== undefined || c.debit !== undefined || c.amount !== undefined);
  });
  if (hi < 0) {
    const c = table.length ? findColumns(table[0]) : {};
    const need = ["date", "balance"].filter((k) => (c as any)[k] === undefined);
    if (c.credit === undefined && c.debit === undefined && c.amount === undefined) need.push("deposit / withdrawal (or amount)");
    throw new Error(`Could not find the transaction table. Missing column${need.length > 1 ? "s" : ""}: ${need.join(", ")}. ` +
      "Export the statement as CSV with Date, Narration, Withdrawal, Deposit and Balance columns.");
  }
  const header = table[hi], col = findColumns(header);
  const rows: StatementRow[] = [];
  let skipped = 0;
  for (const r of table.slice(hi + 1)) {
    const date = parseDate(r[col.date] ?? "");
    const balance = parseAmount(r[col.balance]);
    if (!date || Number.isNaN(balance)) { skipped++; continue; } // footer lines, totals, page headers
    let credit = col.credit !== undefined ? Math.abs(parseAmount(r[col.credit]) || 0) : 0;
    let debit = col.debit !== undefined ? Math.abs(parseAmount(r[col.debit]) || 0) : 0;
    if (col.credit === undefined && col.debit === undefined && col.amount !== undefined) {
      const a = parseAmount(r[col.amount]) || 0, type = (r[col.type] ?? "").toLowerCase();
      if (/cr|credit|deposit/.test(type) || (!type && a > 0)) credit = Math.abs(a); else debit = Math.abs(a);
    }
    if (!credit && !debit) { skipped++; continue; }
    const row: StatementRow = { date, credit, debit, balance };
    if (col.description !== undefined) row.description = r[col.description];
    if (col.category !== undefined) row.category = r[col.category];
    if (col.status !== undefined) row.status = r[col.status];
    rows.push(row);
  }
  if (!rows.length) throw new Error("The file has the right columns but no transactions could be read — check the date and amount formats.");
  const dates = rows.map((r) => r.date).sort();
  const columns = Object.fromEntries(Object.entries(col).map(([k, i]) => [k, header[i]]));
  return { rows, columns, skipped, from: dates[0], to: dates[dates.length - 1] };
}

/** A SYNTHETIC sample written the way a bank exports it, so the upload can be tried end to end. */
export function sampleBankCSV(rows: any[]): string {
  const words: Record<string, string> = { platform_payout: "UPI/CR/DELIVERY PLATFORM WEEKLY PAYOUT", spend: "UPI/DR/FUEL AND FOOD",
    emi: "NACH DR TWO WHEELER LOAN EMI", rent: "IMPS/ROOM RENT", utility: "BILLPAY/ELECTRICITY MSEDCL" };
  const money = (v: number) => (v ? `"${v.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}"` : "");
  const lines = ["SAMPLE BANK STATEMENT — SYNTHETIC DATA, NOT A REAL CUSTOMER", "Account No :,XXXXXXXX4512", "",
    "Date,Narration,Chq./Ref.No.,Withdrawal Amt.,Deposit Amt.,Closing Balance"];
  rows.forEach((r, i) => {
    const [y, m, d] = String(r.date).slice(0, 10).split("-");
    const pre = r.status === "bounced" ? "ACH RETURN INSUFFICIENT FUNDS - " : r.status === "late" ? "LATE PAYMENT FEE - " : "";
    lines.push([`${d}/${m}/${y}`, `"${pre}${words[r.category] || "UPI TRANSFER"} (SYNTHETIC)"`, `REF${String(100000 + i)}`,
      money(r.debit), money(r.credit), money(r.balance)].join(","));
  });
  lines.push("", "*** End of statement ***");
  return lines.join("\r\n");
}
