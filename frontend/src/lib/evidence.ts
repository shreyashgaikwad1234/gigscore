// Current market evidence (published January–October 2026), shared by the Overview and the Market evidence page.
// Every figure is quoted from the named public source; none comes from GigScore's own data. Charts redraw a few
// published data points; no table or chart is reproduced from the reports.
export const SRC = {
  hil: { name: "CRIF High Mark, How India Lends, September 2026 (data to June 2026)", short: "CRIF High Mark, Sept 2026",
    href: "https://www.crifhighmark.com/media/7548/how-india-lends-september-2026.pdf" },
  wheels: { name: "CRIF High Mark, Wheels & Ambition: vehicle finance report, August 2026 (data to June 2026)", short: "CRIF High Mark, Aug 2026",
    href: "https://www.crifhighmark.com/media/7384/wheels-finance-report-august-2026.pdf" },
  cmi: { name: "TransUnion CIBIL, Credit Market Indicator, March 2026 (data to December 2025)", short: "TransUnion CIBIL, Mar 2026",
    href: "https://www.transunioncibil.com/content/dam/transunion-cibil/corporate/documents/cmi-digital-report-mar26.pdf" },
  aa: { name: "CRIF High Mark and Sahamati, Consent to Credit: The Account Aggregator Impact Report, October 2026", short: "CRIF High Mark & Sahamati, Oct 2026",
    href: "https://www.crifhighmark.com/media/7697/account-aggregator-report-october-26.pdf" },
  sahamati: { name: "Sahamati, Account Aggregator crosses 500 million consents, September 2026", short: "Sahamati, Sept 2026",
    href: "https://sahamati.org.in/media-article/account-aggregator-crosses-500-million-consents-as-india-marks-5th-aa-foundation-day/" },
  survey: { name: "Ministry of Finance, Economic Survey 2025-26 (tabled 29 January 2026)", short: "Economic Survey 2025-26",
    href: "https://www.indiabudget.gov.in/economicsurvey/" },
  niti: { name: "NITI Aayog, India's Booming Gig and Platform Economy, June 2022", short: "NITI Aayog, 2022",
    href: "https://www.niti.gov.in/sites/default/files/2022-06/25th_June_Final_Report_27062022.pdf" },
  idinsight: { name: "IDinsight, survey of 2,547 platform delivery riders, 2024", short: "IDinsight, 2024",
    href: "https://www.idinsight.org/publication/economic-lives-of-digital-platform-gig-workers-india/" },
  findex: { name: "World Bank, Global Findex 2025 (survey year 2024)", short: "World Bank Findex 2025",
    href: "https://www.worldbank.org/en/publication/globalfindex" },
} as const;
export type SrcKey = keyof typeof SRC;

// Headline figures: [value, statement, source]
export const HEADLINES: [string, string, SrcKey][] = [
  ["16%", "of new retail loans went to new-to-credit borrowers in late 2025, down from 22% two years earlier. Lenders are favouring applicants with a score.", "cmi"],
  ["80.5%", "of two-wheeler loans go to borrowers new to the product — more than any other vehicle loan. It is how riders enter formal credit.", "wheels"],
  ["0.5% vs 5.3%", "two-wheeler loans 90+ days overdue when the lender used consented bank-statement data, against loans where it did not (June 2026).", "aa"],
  ["1.2 crore", "gig and platform workers in FY25, up 55% in four years. About 40% earn under ₹15,000 a month, and irregular income limits their access to credit.", "survey"],
];

// Strip under the hero: [value, label, source]
export const STRIP: [string, string, SrcKey][] = [
  ["₹1.89 lakh crore", "two-wheeler loans outstanding, June 2026", "hil"],
  ["3.78 crore", "active two-wheeler loans", "hil"],
  ["₹94,967", "average new two-wheeler loan, April–June 2026", "hil"],
  ["500 million", "Account Aggregator consents to share bank data", "sahamati"],
];

export const CHARTS = {
  // New-to-product share of originations (volume), loans availed in the last 12 months
  ntp: { labels: ["Two-wheeler", "Used car", "Car", "Commercial vehicle"], jun25: [81.6, 75.8, 68.5, 56.9], jun26: [80.5, 75.0, 69.4, 55.4] },
  // Share of retail originations by borrower, quarter ended December
  tier: { labels: ["Oct–Dec 2023", "Oct–Dec 2024", "Oct–Dec 2025"], ntc: [22, 18, 16], primePlus: [52, 55, 57], belowPrime: [26, 27, 27] },
  // Two-wheeler delinquency, % of portfolio
  par: { labels: ["Jun 2024", "Jun 2025", "Jun 2026"], p31_90: [3.5, 3.7, 2.8], p91_180: [1.6, 2.1, 1.2] },
  par360: { labels: ["Jun 2025", "Mar 2026", "Jun 2026"], v: [5.8, 5.7, 6.3] },
  // Two-wheeler PAR 31-90 by ticket size, and share of Q1 FY27 origination value
  ticket: { labels: ["₹50K–75K", "₹75K–1L", "₹1L–1.5L", "₹1.5L+"], jun25: [2.89, 4.71, 3.71, 2.69], jun26: [2.20, 3.61, 2.88, 2.13], valueShare: [15.2, 39.8, 28.0, 15.2] },
  // Gig and platform workforce, crore people
  gig: { labels: ["FY21", "FY25", "FY30 (projected)"], v: [0.77, 1.2, 2.35] },
  // PAR 90+ in June 2026, AA-enabled loans vs loans without AA data
  aa: { labels: ["Two-wheeler", "Personal loan", "Business loan ≤ ₹10L", "Used car"], with: [0.5, 2.0, 4.1, 2.4], without: [5.3, 2.2, 4.4, 1.7] },
};
