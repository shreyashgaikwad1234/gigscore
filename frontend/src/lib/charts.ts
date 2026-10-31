// ECharts set-up shared by every page: only the chart types we use, themed from the CSS tokens,
// re-drawn when the theme changes or the window resizes.
import * as echarts from "echarts/core";
import { BarChart, GaugeChart, LineChart, PieChart, SankeyChart, ScatterChart } from "echarts/charts";
import { GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, MarkPointComponent, TitleComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([BarChart, GaugeChart, LineChart, PieChart, SankeyChart, ScatterChart, GridComponent, LegendComponent,
  MarkAreaComponent, MarkLineComponent, MarkPointComponent, TitleComponent, TooltipComponent, CanvasRenderer]);

export { echarts };
export const v = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function palette() {
  return { sky: v("--sky"), sky2: v("--sky-2"), gold: v("--gold"), gold2: v("--gold-2"), mint: v("--mint"), violet: v("--violet"),
    rose: v("--rose"), orange: v("--orange"), text: v("--text"), text2: v("--text-2"), muted: v("--muted"), faint: v("--faint"),
    border: v("--border"), grid: v("--grid"), surface: v("--surface-solid") };
}

export const grad = (top: string, bottom: string) =>
  new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: top }, { offset: 1, color: bottom }]);
export const hgrad = (left: string, right: string) =>
  new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: left }, { offset: 1, color: right }]);
export const alpha = (hex: string, a: number) => {
  if (hex.startsWith("rgba") || hex.startsWith("rgb")) return hex;
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

/** Defaults every chart shares: fonts, axis styling, tooltip look. */
export function base(extra: Record<string, any> = {}) {
  const c = palette();
  const axis = {
    axisLine: { lineStyle: { color: c.border } }, axisTick: { show: false },
    axisLabel: { color: c.muted, fontFamily: "Public Sans", fontSize: 11 },
    splitLine: { lineStyle: { color: c.border, type: [4, 4] } },
    nameTextStyle: { color: c.muted, fontFamily: "Public Sans", fontSize: 11 },
  };
  return {
    backgroundColor: "transparent",
    textStyle: { fontFamily: "Public Sans", color: c.text2 },
    animationDuration: 900, animationEasing: "cubicOut",
    tooltip: { trigger: "axis", backgroundColor: c.surface, borderColor: c.border, borderWidth: 1, padding: [10, 12],
      textStyle: { color: c.text, fontFamily: "Public Sans", fontSize: 12 }, axisPointer: { lineStyle: { color: c.faint }, crossStyle: { color: c.faint } } },
    // full-width legend so long series names wrap instead of being clipped at the card edge
    legend: { textStyle: { color: c.text2, fontSize: 12 }, icon: "roundRect", itemWidth: 12, itemHeight: 4, itemGap: 14,
      top: 0, left: 0, right: 0, align: "left" },
    grid: { left: 8, right: 14, top: 52, bottom: 30, containLabel: true },
    _axis: axis,
    ...extra,
  };
}

const registry: { el: HTMLElement; build: () => any; chart?: echarts.ECharts }[] = [];

/** Mount a chart whose option is rebuilt (with fresh theme colours) on theme change. */
export function mount(el: HTMLElement | null, build: () => any) {
  if (!el) return;
  const existing = registry.find((r) => r.el === el);
  if (existing) { // same element: swap in the new option builder
    existing.build = build;
    existing.chart!.setOption(clean(build()), true);
    return existing.chart;
  }
  const entry = { el, build, chart: echarts.init(el, undefined, { renderer: "canvas" }) };
  entry.chart.setOption(clean(build()));
  registry.push(entry);
  return entry.chart;
}

function clean(o: any) {
  const { _axis, ...rest } = o;
  return rest;
}

export function axisX(extra: any = {}) { return { ...base()._axis, ...extra }; }
export function axisY(extra: any = {}) { return { ...base()._axis, ...extra }; }

window.addEventListener("resize", () => registry.forEach((r) => r.chart?.resize()));
window.addEventListener("themechange", () => registry.forEach((r) => r.chart?.setOption(clean(r.build()), true)));
