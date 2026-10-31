// Semicircular 300-900 score gauge with the decline / refer / approve zones from the live cut-offs.
const MIN = 300, MAX = 900, CX = 120, CY = 120, R = 96;

const angle = (s: number) => Math.PI * (1 - (Math.min(MAX, Math.max(MIN, s)) - MIN) / (MAX - MIN));
const pt = (a: number, r = R) => [CX + r * Math.cos(a), CY - r * Math.sin(a)];

function arc(from: number, to: number) {
  const [x1, y1] = pt(angle(from));
  const [x2, y2] = pt(angle(to));
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${R} ${R} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

export function gaugeSvg(score: number, cuts: { approve: number; refer: number }) {
  const [nx, ny] = pt(angle(score), R - 22);
  const zone = (a: number, b: number, color: string) =>
    `<path d="${arc(a, b)}" stroke="var(${color})" stroke-width="16" fill="none" stroke-linecap="butt" opacity=".9"/>`;
  return `<svg viewBox="0 0 240 132" width="100%" role="img" aria-label="Score ${score} on a 300 to 900 scale" style="max-width:280px">
    <path d="${arc(MIN, MAX)}" stroke="var(--grey-soft)" stroke-width="16" fill="none"/>
    ${zone(MIN, cuts.refer, "--red")}${zone(cuts.refer, cuts.approve, "--amber")}${zone(cuts.approve, MAX, "--green")}
    <line x1="${CX}" y1="${CY}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="var(--text)" stroke-width="3.5" stroke-linecap="round"/>
    <circle cx="${CX}" cy="${CY}" r="7" fill="var(--text)"/>
    <text x="14" y="131" font-size="11" fill="var(--muted)">300</text>
    <text x="206" y="131" font-size="11" fill="var(--muted)">900</text>
  </svg>`;
}
