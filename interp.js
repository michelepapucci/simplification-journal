(() => {
  "use strict";

  // data_interp.js: window.INTERP_G1, built by scripts/build_interp_webview.py from experiments/results/interp_g1.json.
  const D = window.INTERP_G1;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const NF = new Intl.NumberFormat("en-US");
  const state = { model: "g1_guided", view: "in_context", layer: 12, proj: "pca", variant: "within_original", eModel: "g1_guided", example: 0 };

  // ---------- DOM helpers (all text goes through textContent) ----------
  function setAttrs(node, attrs) {
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
  }
  function append(node, children) {
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
  }
  function h(tag, attrs, ...children) { const n = document.createElement(tag); setAttrs(n, attrs); append(n, children); return n; }
  function s(tag, attrs, ...children) { const n = document.createElementNS(SVG_NS, tag); setAttrs(n, attrs); append(n, children); return n; }
  const $ = (id) => document.getElementById(id);
  const clear = (node) => { node.replaceChildren(); return node; };
  const fmt = (x, d = 2) => (x == null ? "–" : (x < 0 ? "−" : "") + Math.abs(x).toFixed(d));
  const signed = (x, d = 2) => (x == null ? "–" : (x < 0 ? "−" : x > 0 ? "+" : "") + Math.abs(x).toFixed(d));
  const pct = (x, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
  const fmtQ = (q) => (q == null ? "–" : q < 0.001 ? q.toExponential(0) : q.toFixed(3));

  // ---------- colour ----------
  let palette = {};
  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const names = ["surface", "text-primary", "text-secondary", "text-muted", "grid", "series-1", "series-2", "series-3", "series-4",
      "series-5", "series-6", "series-7", "series-8", "div-neg", "div-mid", "div-pos"];
    palette = Object.fromEntries(names.map((n) => [n, cs.getPropertyValue(`--${n}`).trim()]));
  }
  const hexToRgb = (hex) => { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((c) => c / 255); };
  const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const toSrgb = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
  function toOklab(hex) {
    const [r, g, b] = hexToRgb(hex).map(toLinear);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const q = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * q, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * q,
      0.0259040371 * l + 0.7827717662 * m - 0.808675766 * q];
  }
  function fromOklab([L, a, b]) {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const q = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * q, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * q,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * q];
    const [r, g, bb] = rgb.map((c) => Math.round(Math.min(1, Math.max(0, toSrgb(c))) * 255));
    return `rgb(${r}, ${g}, ${bb})`;
  }
  const mix = (a, b, t) => { const A = toOklab(a), B = toOklab(b); return fromOklab(A.map((v, i) => v + (B[i] - v) * t)); };
  const divergingColor = (r) => mix(palette["div-mid"], palette[r >= 0 ? "div-pos" : "div-neg"], Math.min(1, Math.abs(r)));

  // ---------- tooltip ----------
  const tip = $("tooltip");
  function showTip(x, y, build) {
    clear(tip); build(tip); tip.hidden = false;
    const pad = 14, r = tip.getBoundingClientRect();
    let left = x + pad, top = y + pad;
    if (left + r.width > innerWidth - 8) left = x - r.width - pad;
    if (top + r.height > innerHeight - 8) top = y - r.height - pad;
    tip.style.left = `${Math.max(8, left)}px`; tip.style.top = `${Math.max(8, top)}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  const tipRow = (color, value, name, dashed) => h("div", { class: "tt-row" },
    color ? h("span", { class: `key-line${dashed ? " dashed" : ""}`, style: `border-color:${color}` }) : null,
    h("strong", { text: value }), h("span", { class: "name", text: name }));

  // ---------- line chart with crosshair ----------
  function niceTicks(lo, hi, count = 4) {
    const raw = (hi - lo) / count, p = 10 ** Math.floor(Math.log10(raw || 1)), f = raw / p;
    const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(Math.abs(v) < 1e-12 ? 0 : v);
    return { ticks, step };
  }
  /** series: [{name, color, values (aligned with xs), dashed}] */
  function lineChart(container, { xs, series, height = 210, xTitle, yFormat = (v) => fmt(v, 1), yLo, yHi, ref, label: aria, xFormat = (v) => v, markX = null, zero = true }) {
    const width = Math.max(220, Math.floor(container.clientWidth || 300));
    const m = { top: 10, right: 12, bottom: 38, left: 46 };
    const vals = series.flatMap((se) => se.values.filter((v) => v != null));
    if (ref) vals.push(...ref.values.filter((v) => v != null));
    let lo = zero ? Math.min(0, ...vals) : Math.min(...vals), hi = zero ? Math.max(0, ...vals) : Math.max(...vals);
    if (yLo != null) lo = Math.min(lo, yLo);
    if (yHi != null) hi = Math.max(hi, yHi);
    const padY = (hi - lo) * 0.06 || 1; lo -= padY; hi += padY;
    const x0 = xs[0], x1 = xs[xs.length - 1];
    const x = (v) => m.left + ((v - x0) / (x1 - x0 || 1)) * (width - m.left - m.right);
    const y = (v) => height - m.bottom - ((v - lo) / (hi - lo)) * (height - m.top - m.bottom);
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": aria, tabindex: 0 });
    const { ticks, step } = niceTicks(lo, hi, 4);
    for (const v of ticks) {
      svg.append(s("line", { class: v === 0 ? "baseline" : "gridline", x1: m.left, x2: width - m.right, y1: y(v), y2: y(v) }));
      svg.append(s("text", { class: "axis-text", x: m.left - 6, y: y(v) + 4, "text-anchor": "end", text: yFormat(v, step) }));
    }
    const every = Math.ceil(xs.length / Math.max(2, Math.floor((width - m.left - m.right) / 34)));
    xs.forEach((v, i) => { if (i % every === 0 || i === xs.length - 1) svg.append(s("text", { class: "axis-text", x: x(v), y: height - m.bottom + 15, "text-anchor": "middle", text: xFormat(v) })); });
    if (xTitle) svg.append(s("text", { class: "axis-title", x: m.left + (width - m.left - m.right) / 2, y: height - 4, "text-anchor": "middle", text: xTitle }));
    if (markX != null) svg.append(s("line", { x1: x(markX), x2: x(markX), y1: m.top, y2: height - m.bottom, stroke: palette["text-secondary"], "stroke-width": 1, "stroke-dasharray": "3 3" }));
    const path = (values) => {
      let d = "", pen = false;
      values.forEach((v, i) => { if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`; pen = true; });
      return d;
    };
    if (ref) svg.append(s("path", { d: path(ref.values), fill: "none", stroke: palette["text-muted"], "stroke-width": 1.5, "stroke-dasharray": "2 4", "stroke-linecap": "round" }));
    for (const se of series) {
      svg.append(s("path", { d: path(se.values), fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round",
        "stroke-dasharray": se.dashed ? "6 4" : null }));
    }
    const hover = s("g", { style: "display:none" });
    const cross = s("line", { class: "crosshair", y1: m.top, y2: height - m.bottom });
    hover.append(cross);
    const dots = series.map((se) => { const c = s("circle", { r: 4, fill: se.color, stroke: palette.surface, "stroke-width": 2 }); hover.append(c); return c; });
    svg.append(hover);
    const overlay = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent" });
    svg.append(overlay);
    let idx = 0;
    function show(i, cx, cy) {
      idx = Math.max(0, Math.min(xs.length - 1, i));
      hover.style.display = "";
      cross.setAttribute("x1", x(xs[idx])); cross.setAttribute("x2", x(xs[idx]));
      series.forEach((se, k) => {
        const v = se.values[idx];
        dots[k].style.display = v == null ? "none" : "";
        if (v != null) { dots[k].setAttribute("cx", x(xs[idx])); dots[k].setAttribute("cy", y(v)); }
      });
      showTip(cx, cy, (t) => {
        t.append(h("div", { class: "tt-title", text: xTitle ? `${xTitle}: ${xFormat(xs[idx])}` : String(xs[idx]) }));
        series.forEach((se) => t.append(tipRow(se.color, se.values[idx] == null ? "–" : yFormat(se.values[idx], step / 10), se.name, se.dashed)));
        if (ref && ref.values[idx] != null) t.append(tipRow(palette["text-muted"], yFormat(ref.values[idx], step / 10), ref.name, true));
      });
    }
    const hide = () => { hover.style.display = "none"; hideTip(); };
    overlay.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const v = x0 + ((e.clientX - r.left - m.left) / (width - m.left - m.right)) * (x1 - x0);
      let best = 0;
      xs.forEach((xv, i) => { if (Math.abs(xv - v) < Math.abs(xs[best] - v)) best = i; });
      show(best, e.clientX, e.clientY);
    });
    overlay.addEventListener("pointerleave", hide);
    svg.addEventListener("keydown", (e) => {
      if (e.key === "Escape") return hide();
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const r = svg.getBoundingClientRect(), next = Math.max(0, Math.min(xs.length - 1, idx + (e.key === "ArrowRight" ? 1 : -1)));
      show(next, r.left + x(xs[next]), r.top + m.top);
    });
    svg.addEventListener("blur", hide);
    container.append(svg);
  }
  function legend(container, items) {
    clear(container);
    for (const it of items) {
      container.append(h("span", { class: "legend-item" }, h("span", { class: `key-line${it.dashed ? " dashed" : ""}`, style: `border-color:${it.color}` }), it.name));
    }
  }


  // ---------- page state ----------
  const N = 9;                                          // levels 0-80 (90 is never a training target; "no level" only in C)
  const LV = [0, 10, 20, 30, 40, 50, 60, 70, 80];
  const VIEWS = { in_context: "In context", alone: "Alone" };
  const PROJ = { pca: "PCA", mds: "MDS", tsne: "t-SNE", umap: "UMAP" };
  const VARIANTS = { as_is: "As is", within_original: "Within original" };
  const MODEL_LABEL = Object.fromEntries((D ? D.meta.models : []).map((m) => [m.id, m.label]));
  const levelColor = (i) => mix(palette.surface, palette["series-1"], 0.28 + 0.72 * (i / (N - 1)));
  const geo = () => D.models[state.model].geometry;
  const layerData = () => geo().layers[state.layer][state.view];
  const projLayer = () => Math.min(28, Math.max(2, Math.round(state.layer / 2) * 2));
  const times = (x, d = 2) => (x == null ? "–" : `×${x.toFixed(d)}`);

  function segmented(id, options, value, onPick) {
    const box = clear($(id));
    for (const [v, text] of Object.entries(options)) {
      box.append(h("button", { type: "button", role: "radio", "data-value": v, "aria-checked": String(v === value), text,
        onclick: () => { onPick(v); box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === v))); } }));
    }
  }

  // ---------- A: the level map at one layer ----------
  function mapPoints() {
    // returns {means: [[x, y]] for levels 0-80 or null, cloud: [{p: [x, y], i}] , note}
    if (state.proj === "pca") {
      const L = layerData();
      if (L.rho_pc1 == null) return { empty: "At this layer the state where generation starts has not yet read the control token: every level gives the same state." };
      const cloud = state.view === "in_context" ? L.cloud.flatMap((row) => row.map((p, i) => ({ p, i }))) : [];
      return { means: L.coords, cloud, xTitle: `Main direction (PC1, ${pct(L.pc1_share)} of the spread between level means)` };
    }
    if (!D.projections) return { empty: "Projections not available." };
    const pl = projLayer(), P = D.projections.models[state.model].layers[String(pl)];
    const note = pl !== state.layer ? `MDS, t-SNE and UMAP are computed every second layer: showing layer ${pl}.` : "";
    if (state.proj === "mds") return { means: P[state.view === "in_context" ? "mds_in_context" : "mds_alone"], cloud: [], note,
      xTitle: "MDS of the 9 level means (distances preserved as far as possible)" };
    if (state.view === "alone") return { empty: "t-SNE and UMAP need many points: they are shown for single originals in context only." };
    const pts = P[`${state.proj}_${state.variant}`];
    const cloud = pts.map((p, k) => ({ p, i: k % N }));
    const means = LV.map((_, i) => {
      const sel = cloud.filter((c) => c.i === i);
      return [sel.reduce((a, c) => a + c.p[0], 0) / sel.length, sel.reduce((a, c) => a + c.p[1], 0) / sel.length];
    });
    return { means, cloud, note, centroids: true,
      xTitle: `${PROJ[state.proj]} of ${D.projections.n_originals} originals × 9 levels (${VARIANTS[state.variant].toLowerCase()}); large dots: each level’s centre` };
  }

  function levelMap(container) {
    clear(container);
    const M = mapPoints();
    if (M.empty) { container.append(h("p", { class: "sub", text: M.empty })); return; }
    const width = Math.max(260, Math.floor(container.clientWidth || 480)), height = Math.min(420, Math.max(260, Math.round(width * 0.75)));
    const m = { top: 14, right: 18, bottom: 34, left: 18 };
    const all = [...M.means, ...M.cloud.map((c) => c.p)].filter((p) => p && p[0] != null);
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.08 || 1;
    const x = (v) => m.left + ((v - x0 + px) / (x1 - x0 + 2 * px)) * (width - m.left - m.right);
    const y = (v) => height - m.bottom - ((v - y0 + py) / (y1 - y0 + 2 * py)) * (height - m.top - m.bottom);
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: 0,
      "aria-label": `Levels 0 to 80 at layer ${state.layer}, ${PROJ[state.proj]}` });
    svg.append(s("line", { class: "baseline", x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }));
    svg.append(s("text", { class: "axis-title", x: m.left + (width - m.left - m.right) / 2, y: height - 8, "text-anchor": "middle", text: M.xTitle }));
    const small = M.cloud.length > 1000;
    for (const c of M.cloud) svg.append(s("circle", { cx: x(c.p[0]), cy: y(c.p[1]), r: small ? 1.6 : 2, fill: levelColor(c.i), opacity: small ? 0.55 : 0.35 }));
    svg.append(s("path", { d: `M${M.means.map((p) => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("L")}`,
      fill: "none", stroke: palette["text-secondary"], "stroke-width": 1.5, "stroke-linejoin": "round" }));
    M.means.forEach((p, i) => {
      const cx = x(p[0]), cy = y(p[1]);
      svg.append(s("circle", { cx, cy, r: 7, fill: levelColor(i), stroke: palette.surface, "stroke-width": 2 }));
      svg.append(s("text", { class: "axis-text", x: cx + 9, y: cy - 8, text: String(LV[i]) }));
      const hit = s("circle", { cx, cy, r: 12, fill: "transparent" });
      hit.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, (t) => {
        t.append(h("div", { class: "tt-title", text: `Level ${LV[i]} · layer ${state.proj === "pca" ? state.layer : projLayer()}` }));
        if (M.centroids) t.append(h("div", { class: "tt-note", text: "Centre of this level’s single-original points" }));
      }));
      hit.addEventListener("pointerleave", hideTip);
      svg.append(hit);
    });
    container.append(svg);
    if (M.note) container.append(h("p", { class: "note", text: M.note }));
  }

  function cosineMatrix(container) {
    clear(container);
    const C = layerData().cosine;
    if (!C || C[0][0] == null) return;
    const t = h("table", { class: "data cosine" });
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "" }), LV.map((n) => h("th", { class: "num", text: n })))));
    const body = h("tbody");
    C.forEach((row, i) => body.append(h("tr", {}, h("th", { scope: "row", text: LV[i] }), row.map((v, j) => h("td", {
      class: "num", style: `background:${divergingColor(-v)};color:${Math.abs(v) > 0.6 ? palette.surface : palette["text-primary"]}`,
      title: `level ${LV[i]} vs level ${LV[j]}: ${fmt(v, 2)}`, text: fmt(v, 1) })))));
    t.append(body);
    container.append(t);
  }

  function overLayers(container) {
    clear(container);
    const ls = geo().layers.map((l) => l[state.view]), xs = ls.map((_, i) => i);
    const series = [
      { name: "Order along the main direction (Spearman ρ)", color: palette["series-1"], values: ls.map((l) => l.rho_pc1) },
      { name: "In-between levels in between (share of triples)", color: palette["series-2"], values: ls.map((l) => l.betweenness) },
      { name: "Clustering merges neighbours (share)", color: palette["series-3"], values: ls.map((l) => l.contiguous_merges) },
    ];
    if (state.view === "in_context") series.push({ name: "Requested level read back (cross-validated R²)", color: palette["series-4"], values: ls.map((l) => l.readout_r2) });
    lineChart(container, { xs, series, height: 230, xTitle: "Layer", yFormat: (v) => fmt(v, 2), yLo: 0, yHi: 1, markX: state.layer,
      label: "Ordering and hierarchy measures of levels 0 to 80 at every layer" });
    return series;
  }

  function spacingChart(container) {
    clear(container);
    const sp = layerData().spacing;
    if (!sp || sp[0] == null) return [];
    const beh = D.behaviour[state.model] || {};
    const gaps = [0, 10, 20, 30, 40, 50, 60].map((lv) => (beh[lv + 10] != null && beh[lv] != null ? beh[lv + 10] - beh[lv] : null));
    const ok = gaps.filter((v) => v != null), meanGap = ok.reduce((a, b) => a + b, 0) / ok.length;
    const xs = [5, 15, 25, 35, 45, 55, 65, 75];
    const series = [
      { name: `Distance between consecutive levels in the model, layer ${state.layer}`, color: palette["series-1"], values: sp },
      { name: "Distance between their outputs’ mean READ-IT", color: palette["series-2"], values: [...gaps.map((g) => (g == null ? null : g / meanGap)), null], dashed: true },
    ];
    lineChart(container, { xs, series, height: 210, xTitle: "Step between levels", xFormat: (v) => `${v - 5}–${v + 5}`, yFormat: (v) => fmt(v, 1),
      yLo: 0, ref: { name: "even spacing", values: xs.map(() => 1) }, label: "Spacing between consecutive levels, relative to the mean step" });
    return series;
  }

  function renderA() {
    $("layer-label").textContent = `Layer ${state.layer}${state.layer === 0 ? " (embeddings)" : ""}`;
    $("layer").value = String(state.layer);
    $("a-variant").hidden = !(state.proj === "tsne" || state.proj === "umap");
    const L = layerData();
    $("a-stats").textContent = L.rho_pc1 == null ? "" :
      `Layer ${state.layer}, ${VIEWS[state.view].toLowerCase()}: order ρ ${fmt(L.rho_pc1, 2)}, in-between ${pct(L.betweenness)}, neighbour merges ${pct(L.contiguous_merges)}` +
      (L.readout_r2 != null ? `, level read back R² ${fmt(L.readout_r2, 3)}.` : ".");
    levelMap($("a-map"));
    cosineMatrix($("a-cosine"));
    legend($("a-over-legend"), overLayers($("a-over")));
    legend($("a-spacing-legend"), spacingChart($("a-spacing")));
    const e = geo().embeddings.input;
    $("a-emb").textContent = `Input embeddings of levels 0–80 (${MODEL_LABEL[state.model]}): order ρ ${fmt(e.rho_pc1, 2)}, in-between ${pct(e.betweenness)}, ` +
      `neighbour merges ${pct(e.contiguous_merges)}. The output layer never predicts these tokens, so its rows keep their initial values.`;
  }

  // ---------- B: readout ----------
  function renderB() {
    const xs = D.models[D.meta.models[0].id].readout.state_by_layer.map((_, i) => i);
    const series = D.meta.models.map((m, k) => ({ name: `${m.label}: hidden state at that layer`, color: palette[k ? "series-2" : "series-1"],
      values: D.models[m.id].readout.state_by_layer }));
    const r0 = D.models[D.meta.models[0].id].readout;
    series.push({ name: "The request alone", color: palette["text-secondary"], values: xs.map(() => r0.request), dashed: true });
    series.push({ name: "The request and the original’s READ-IT and length", color: palette["series-3"], values: xs.map(() => r0.request_and_original), dashed: true });
    lineChart($("b-chart"), { xs, series, height: 230, xTitle: "Layer", yFormat: (v) => fmt(v, 2), yLo: 0, yHi: 0.35,
      label: "How well each predictor explains the READ-IT the output reaches" });
    legend($("b-legend"), series);
  }

  // ---------- C: no level ----------
  function renderC() {
    const n = D.no_level;
    if (!n) return;
    const lv10 = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
    const o = n.outputs;
    const series = [
      { name: "“No level”", color: palette["series-4"], values: lv10.map((l) => o["no level"].shares[l] ?? 0) },
      { name: "Request 30", color: palette["series-1"], values: lv10.map((l) => o["30"].shares[l] ?? 0), dashed: true },
      { name: "Request 40", color: palette["series-3"], values: lv10.map((l) => o["40"].shares[l] ?? 0) },
      { name: "Request 50", color: palette["series-2"], values: lv10.map((l) => o["50"].shares[l] ?? 0), dashed: true },
    ];
    lineChart($("c-out"), { xs: lv10, series, height: 210, xTitle: "READ-IT level reached", yFormat: (v) => pct(v), yLo: 0,
      label: "Where the no-level outputs land, against requests 30, 40 and 50 on the same originals" });
    legend($("c-out-legend"), series);
    $("c-out-note").textContent = `Means: no level ${fmt(o["no level"].mean, 1)}, request 30 ${fmt(o["30"].mean, 1)}, 40 ${fmt(o["40"].mean, 1)}, 50 ${fmt(o["50"].mean, 1)}. ` +
      `Distance between the distributions (Kolmogorov–Smirnov): to request 30 ${fmt(o["30"].ks_vs_no_level, 3)}, 40 ${fmt(o["40"].ks_vs_no_level, 3)}, 50 ${fmt(o["50"].ks_vs_no_level, 3)}.`;
    const lv8 = [0, 10, 20, 30, 40, 50, 60, 70];
    const s2 = [
      { name: "Similarity of the no-level output to that level’s output (mean chrF)", color: palette["series-4"], values: lv8.map((l) => n.chrf_by_level[l]) },
    ];
    lineChart($("c-chrf"), { xs: lv8, series: s2, height: 200, xTitle: "Requested level", yFormat: (v) => fmt(v, 0),
      label: "Character n-gram similarity between the no-level output and each level's output on the same original" });
    legend($("c-chrf-legend"), s2);
    const cl = n.closest_level_share_levels_0_60;
    $("c-chrf-note").textContent = "Share of originals (with every level 0–60) whose most similar output is that level’s: " +
      Object.entries(cl).map(([k, v]) => `${k}: ${pct(v)}`).join(", ") + ". By chance, 14% each.";
    const st = n.state_by_layer, xs = st.map((_, i) => i);
    const s3 = [{ name: "“No level” read as a level (position on the main direction)", color: palette["series-4"], values: st.map((x) => (x ? x.equivalent_level : null)) }];
    lineChart($("c-state"), { xs, series: s3, height: 200, xTitle: "Layer", yFormat: (v) => fmt(v, 0), yLo: 0, yHi: 80,
      ref: { name: "level 40", values: xs.map(() => 40) }, label: "Where the no-level state sits on the level direction, by layer" });
    legend($("c-state-legend"), [...s3, { name: "Level 40", color: palette["text-muted"], dashed: true }]);
    const l12 = st[12];
    const s4 = [{ name: "Distance from “no level” to each level, layer 12 (in steps between consecutive levels)", color: palette["series-4"], values: l12.distance_in_steps }];
    lineChart($("c-dist"), { xs: LV, series: s4, height: 200, xTitle: "Level", yFormat: (v) => fmt(v, 1), yLo: 0,
      label: "Distance from the no-level state to each level's state at layer 12" });
    legend($("c-dist-legend"), s4);
    const nearest = st.filter(Boolean).map((x) => x.nearest);
    $("c-state-note").textContent = `Nearest level: ${nearest.every((v) => v === 40) ? "40 at every layer" : nearest.join(", ")}.`;
  }

  // ---------- D: vocabulary ----------
  const COND = { g1: ["g1", "series-1", false], g1_guided: ["g1 · guided, w = 1", "series-3", false], g1_guided_w3: ["g1 · guided, w = 3", "series-3", true] };
  function renderD() {
    const V = D.vocabulary, levels = [0, 10, 20, 30, 40, 50, 60, 70];
    const byLevel = (cond, key) => levels.map((lv) => { const b = V.generated[cond] && V.generated[cond].by_level[lv]; return b ? b[key] : null; });
    const mk = (key) => Object.entries(COND).filter(([c]) => V.generated[c]).map(([c, [name, col, dashed]]) => ({ name, color: palette[col], dashed, values: byLevel(c, key) }));
    const pf = (v, step) => `${(100 * v).toFixed(step < 0.01 ? 1 : 0)}%`;
    const s1 = mk("basic_share");
    lineChart($("d-dist"), { xs: levels, series: s1, height: 220, xTitle: "Requested level", yFormat: pf, zero: false, yLo: 0.85, yHi: 0.95,
      label: "Share of the next-token probability on De Mauro basic vocabulary, by requested level" });
    legend($("d-dist-legend"), s1);
    const s2 = mk("basic_share_text");
    s2.push({ name: "IMPaCTS references at that level (test)", color: palette["series-2"], values: levels.map((lv) => (V.references_text_by_level[lv] || {}).basic_share_text ?? null) });
    s2.push({ name: "g1 training simplifications at that level", color: palette["series-5"], values: levels.map((lv) => (V.training_text_by_level[lv] || {}).basic_share_text ?? null) });
    lineChart($("d-text"), { xs: levels, series: s2, height: 220, xTitle: "Requested / target level", yFormat: pf, zero: false, yLo: 0.85, yHi: 0.95,
      ref: { name: "The test originals", values: levels.map(() => V.originals_text.basic_share_text) },
      label: "Share of words in De Mauro basic vocabulary, by level: outputs, references and training data" });
    legend($("d-text-legend"), [...s2, { name: "The test originals", color: palette["text-muted"], dashed: true }]);
    const toks = D.meta.levels;
    const s3 = D.meta.models.filter((m) => V.fixed[m.id]).map((m, k) => ({ name: m.label, color: palette[k ? "series-1" : "series-3"],
      values: toks.map((t) => (V.fixed[m.id].by_token[t] ? V.fixed[m.id].by_token[t].basic_share : null)) }));
    lineChart($("d-fixed"), { xs: toks.map(Number), series: s3, height: 220, xTitle: "Control token (level)", yFormat: pf, zero: false, yLo: 0.85, yHi: 0.95,
      label: "Share of the next-token probability on basic vocabulary, the same text read under each token" });
    legend($("d-fixed-legend"), s3);
  }

  // ---------- E: word classes ----------
  const E_LV = ["40", "30", "20", "10", "0"];
  const CLASS_LABEL = { NOUN: "Nouns", PROPN: "Proper nouns", VERB: "Verbs", AUX: "Auxiliaries and copulas", ADJ: "Adjectives", ADV: "Adverbs",
    DET: "Determiners", ADP: "Adpositions", PRON: "Pronouns", CCONJ: "Coordinating conjunctions", SCONJ: "Subordinating conjunctions",
    NUM: "Numerals", "full stop": "Full stops", comma: "Commas", "semicolon or colon": "Semicolons and colons", "other punctuation": "Other punctuation", other: "Other" };
  const E = () => D.classes[state.eModel];
  function ratioBar(ratio, ci, strong) {
    const lr = Math.log(ratio), w = Math.min(50, Math.abs(lr) / Math.log(2) * 50);   // a halving or doubling fills half the cell
    const col = lr >= 0 ? palette["div-neg"] : palette["div-pos"];
    return h("td", { class: `num ratio${strong ? " strong" : ""}`, title: ci ? `95% interval ×${ci[0].toFixed(2)}–×${ci[1].toFixed(2)}` : null },
      h("span", { class: "rbar-wrap" }, h("span", { class: "rbar", style: `${lr >= 0 ? "left:50%" : `right:50%`};width:${w}%;background:${col}` })),
      h("span", { class: "rval", text: times(ratio) }));
  }
  function renderE() {
    const e = E();
    if (!e) return;
    const rows = Object.entries(e.words).sort((a, b) => a[1]["0"].ratio - b[1]["0"].ratio);
    const t = h("table", { class: "data eratio" });
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "Class of the text’s word" }), h("th", { class: "num", text: "Words" }),
      ["30", "20", "10", "0"].map((lv) => h("th", { class: `num${lv === "0" || lv === "20" ? " strong" : ""}`, text: `Level ${lv}` })))));
    const body = h("tbody");
    for (const [c, v] of rows) {
      body.append(h("tr", {}, h("td", { text: CLASS_LABEL[c] || c }), h("td", { class: "num", text: NF.format(v.words) }),
        ["30", "20", "10", "0"].map((lv) => ratioBar(v[lv].ratio, v[lv].ci95, lv === "0" || lv === "20"))));
    }
    t.append(body);
    clear($("e-words")).append(t);
    const lv5 = [0, 10, 20, 30, 40];
    const st = e.steps;
    const sc = [["end of text", "End of text", "series-4"], ["full stop", "Full stop", "series-1"], ["comma", "Comma", "series-2"], ["semicolon or colon", "Semicolon or colon", "series-3"]]
      .map(([k, name, col]) => ({ name, color: palette[col], values: lv5.map((l) => st[k][String(l)] / st[k]["40"]) }));
    lineChart($("e-steps"), { xs: lv5, series: sc, height: 220, xTitle: "Level the fixed text is read under", yFormat: (v) => `×${fmt(v, 1)}`, yLo: 0,
      ref: { name: "level 40", values: lv5.map(() => 1) }, label: "Probability at every step of ending the text or of punctuation, relative to level 40" });
    legend($("e-steps-legend"), sc);
    const end = st["end of text, at the text's end"];
    $("e-steps-note").textContent = `Per step at level 40: end of text ${pct(st["end of text"]["40"], 2)}, full stop ${pct(st["full stop"]["40"], 1)}, comma ${pct(st.comma["40"], 1)}, ` +
      `semicolon or colon ${pct(st["semicolon or colon"]["40"], 2)}. Ending exactly where the text ends: ${pct(end["40"], 1)} at level 40, ${pct(end["20"], 1)} at 20, ${pct(end["0"], 1)} at 0.`;
    const tok = (list) => {
      const tt = h("table", { class: "data pushed" });
      tt.append(h("thead", {}, h("tr", {}, h("th", { text: "Item" }), ["40", "20", "0"].map((lv) => h("th", { class: "num", text: `P at ${lv}` })), h("th", { class: "num", text: "0 vs 40" }))));
      const b = h("tbody");
      for (const x of list) b.append(h("tr", {}, h("td", {}, h("code", { text: JSON.stringify(x.item).slice(1, -1) })),
        ["40", "20", "0"].map((lv) => h("td", { class: "num", text: pct(x.prob[lv], 2) })), h("td", { class: "num", text: times(x.ratio_0_vs_40) })));
      tt.append(b);
      return tt;
    };
    const box = clear($("e-tokens"));
    box.append(h("div", { class: "multiple" }, h("h4", { text: "Up, step by step from 40 to 0" }), h("div", { class: "scroll-x" }, tok(e.tokens.up))),
      h("div", { class: "multiple" }, h("h4", { text: "Down, step by step from 40 to 0" }), h("div", { class: "scroll-x" }, tok(e.tokens.down))));
    $("e-tokens-note").textContent = `Of the ${e.tokens.n_items_considered} items with at least 0.1% mean probability at some level, ${e.tokens.n_monotone} change in the same direction at every step from 40 to 0; the ten that change most each way.`;
    const pos = ["AUX", "PRON", "ADV", "ADP", "SCONJ", "CCONJ", "PROPN", "DET", "VERB", "NOUN", "ADJ", "PUNCT"];
    const lv8 = ["0", "10", "20", "30", "40", "50", "60", "70"];
    const grid = clear($("e-texts"));
    for (const p of pos) {
      const div = h("div", { class: "multiple" }, h("h4", { text: CLASS_LABEL[p] || "Punctuation" }));
      const chart = h("div", { class: "chart" });
      div.append(chart);
      grid.append(div);
      const ser = [
        { name: "Outputs", color: palette["series-1"], values: lv8.map((l) => (e.texts.outputs[l] ? e.texts.outputs[l][p] / 100 : null)) },
        { name: "IMPaCTS references", color: palette["series-2"], values: lv8.map((l) => (e.texts.references[l] ? e.texts.references[l][p] / 100 : null)), dashed: true },
      ];
      lineChart(chart, { xs: lv8.map(Number), series: ser, height: 150, xTitle: "Level", yFormat: (v, step) => `${(100 * v).toFixed(step < 0.01 ? 1 : 0)}%`,
        ref: { name: "Originals", values: lv8.map(() => e.texts.originals[p] / 100) }, label: `Share of ${p} words by level` });
    }
    legend($("e-texts-legend"), [{ name: "Outputs of the selected model at that requested level", color: palette["series-1"] },
      { name: "IMPaCTS references at that level", color: palette["series-2"], dashed: true }, { name: "Test originals", color: palette["text-muted"], dashed: true }]);
    renderExample();
  }
  function renderExample() {
    const e = E(), ex = e.examples[Math.min(state.example, e.examples.length - 1)];
    const box = clear($("e-example"));
    box.append(h("p", { class: "example-original" }, h("span", { class: "tag", text: "Original" }), ex.original));
    const d0 = ex.levels["0"].word_logp.map((v, i) => v - ex.levels["40"].word_logp[i]);
    const words = h("p", { class: "example-words" }, h("span", { class: "tag", text: "Level-40 output: each word coloured by how much more (blue) or less (red) probable it is at level 0" }));
    ex.words.forEach((w, i) => {
      const r = Math.max(-1, Math.min(1, d0[i] / Math.log(4)));
      words.append(h("span", { class: "w", style: `background:${divergingColor(-r)}`, title: `${w.text} (${CLASS_LABEL[w.class] || w.class}): ×${Math.exp(d0[i]).toFixed(2)} at level 0 vs 40` }, w.text), " ");
    });
    box.append(words);
    const steps = ex.tokens.map((_, i) => i);
    const lab = (i) => JSON.stringify(ex.tokens[i]).slice(1, -1);
    const mkS = (k) => [["40", "series-3", false], ["20", "series-1", true], ["0", "series-4", false]].map(([lv, col, dashed]) => ({ name: `Level ${lv}`, color: palette[col], dashed, values: ex.levels[lv].steps[k] }));
    const c1 = h("div", { class: "chart" }), c2 = h("div", { class: "chart" });
    box.append(h("div", { class: "multiples two" },
      h("div", { class: "multiple" }, h("h4", { text: "Probability of ending the text here" }), c1),
      h("div", { class: "multiple" }, h("h4", { text: "Probability of a full stop here" }), c2)));
    const legendBox = h("div", { class: "legend" });
    box.append(legendBox);
    lineChart(c1, { xs: steps, series: mkS("end of text"), height: 190, xTitle: "Step (hover for the token that follows)", xFormat: (i) => lab(i), yFormat: (v) => pct(v, 0), yLo: 0, label: "Probability of ending at each step" });
    lineChart(c2, { xs: steps, series: mkS("full stop"), height: 190, xTitle: "Step (hover for the token that follows)", xFormat: (i) => lab(i), yFormat: (v) => pct(v, 0), yLo: 0, label: "Probability of a full stop at each step" });
    legend(legendBox, mkS("end of text"));
  }

  // ---------- F: attention and syntax ----------
  const REL = { obj: "object", nsubj: "subject", conj: "conjunct", acl: "clause modifying a noun", advcl: "adverbial clause", ccomp: "clausal complement",
    xcomp: "open complement", obl: "oblique", nmod: "noun modifier", amod: "adjective", appos: "apposition", parataxis: "parataxis", nummod: "numeral",
    compound: "compound", flat: "flat name", fixed: "fixed expression", advmod: "adverb" };
  const IDX = { ent: "Entropy of the syntax heads’ attention", dist: "Expected attention distance (units)", long: "Share of attention more than 5 units back" };
  function renderF() {
    const A = D.attention;
    if (!A) return;
    const m = A.models.g1_guided, sh = m.syntax_heads;
    $("f-heads-note").textContent = `${sh.n_heads} syntax heads in g1 · guided (${A.models.g1.syntax_heads.n_heads} in g1, ${A.head_overlap.both} shared): ` +
      `${sh.confirmed_pairs} head–relation pairs beat the relation’s best fixed offset by at least 10 points on the held-out texts, out of ${sh.candidates} selected on the other half.`;
    const t = h("table", { class: "data" });
    t.append(h("thead", {}, h("tr", {}, ["Relation (the word attends to its head)", "Best head", "Held-out accuracy", "Best fixed offset", "Arcs"].map((x, i) => h("th", { class: i > 1 ? "num" : null, text: x })))));
    const b = h("tbody");
    Object.entries(sh.best_per_relation).sort((x, y) => (y[1].acc_heldout - y[1].baseline_heldout) - (x[1].acc_heldout - x[1].baseline_heldout)).forEach(([rel, v]) => {
      b.append(h("tr", {}, h("td", { text: `${REL[rel] || rel} (${rel})` }), h("td", { text: `layer ${v.layer}, head ${v.index}` }),
        h("td", { class: "num" }, h("span", { class: "in", text: pct(v.acc_heldout) })), h("td", { class: "num", text: pct(v.baseline_heldout) }), h("td", { class: "num", text: NF.format(v.arcs_heldout) })));
    });
    t.append(b);
    clear($("f-heads")).append(t);
    const layers = Array.from({ length: 28 }, (_, i) => i);
    const s1 = [{ name: "Syntax heads in the layer (of 16)", color: palette["series-1"], values: layers.map((l) => sh.heads_per_layer[l] || 0) }];
    lineChart($("f-layers"), { xs: layers, series: s1, height: 170, xTitle: "Layer", yFormat: (v) => fmt(v, 0), yLo: 0, label: "Number of syntax heads per layer" });
    legend($("f-layers-legend"), s1);
    const V = A.validation, targets = { readit_syntax: "READ-IT syntax", readit_global: "READ-IT global", parse_mean_dep_len: "Dependency length",
      parse_mean_max_depth: "Tree depth", subordinate_proposition_dist: "Subordination" };
    const cols = Object.keys(V);
    const vt = h("table", { class: "data" });
    vt.append(h("thead", {}, h("tr", {}, h("th", { text: "Index" }), Object.values(targets).map((x) => h("th", { class: "num", text: x })), h("th", { class: "num", text: "Length" }))));
    const vb = h("tbody");
    for (const c of cols) {
      const label = c.startsWith("probe_len") ? `Probe: dependency length (layer ${c.split("_L")[1]})` : { ent: "Entropy", dist: "Attention distance", long: "Long-range share" }[c];
      vb.append(h("tr", { class: c === "ent" ? "selected" : null }, h("td", { text: label }),
        Object.keys(targets).map((k) => h("td", { class: "num" }, `${fmt(V[c][k].rho, 2)} `, h("span", { class: "muted", text: `(${fmt(V[c][k].partial_length, 2)})` }))),
        h("td", { class: "num", text: fmt(V[c].length.rho, 2) })));
    }
    vt.append(vb);
    clear($("f-valid")).append(vt);
    const P = A.probe;
    if (P) {
      const ls = Object.keys(P.dev_uuas_best_epoch).map(Number);
      const s2 = [{ name: "Probe: undirected attachment score on the dev sentences", color: palette["series-3"], values: ls.map((l) => P.dev_uuas_best_epoch[l]) }];
      lineChart($("f-probe"), { xs: ls, series: s2, height: 170, xTitle: "Layer", yFormat: (v) => fmt(v, 2), yLo: 0, yHi: 1, label: "Structural probe accuracy by layer" });
      legend($("f-probe-legend"), s2);
      $("f-probe-note").textContent = P.test_uuas ? `Best layer ${P.best_layer}; on the held-out test texts UUAS ${fmt(P.test_uuas.original, 2)} (originals) and ${fmt(P.test_uuas.reference, 2)} (references). ` +
        `Its predicted dependency length against the syntax heads’ entropy: ρ ${fmt(P.attention_vs_probe.ent, 2)}.` : "";
    }
    const S = A.step_2_1, lv8 = [0, 10, 20, 30, 40, 50, 60, 70];
    const c1 = [{ name: IDX.ent, color: palette["series-1"], values: lv8.map((l) => S.by_level[l].ent) }];
    lineChart($("f-21-ent"), { xs: lv8, series: c1, height: 190, xTitle: "Requested level", yFormat: (v) => fmt(v, 3), zero: false, label: "Entropy of the syntax heads by requested level" });
    legend($("f-21-ent-legend"), c1);
    const c2 = [{ name: IDX.dist, color: palette["series-2"], values: lv8.map((l) => S.by_level[l].dist) }];
    if (S.by_level[0].probe_len != null) c2.push({ name: "Probe: predicted dependency length", color: palette["series-3"], values: lv8.map((l) => S.by_level[l].probe_len), dashed: true });
    lineChart($("f-21-dist"), { xs: lv8, series: c2, height: 190, xTitle: "Requested level", yFormat: (v) => fmt(v, 1), yLo: 0, label: "Attention distance and probe dependency length by requested level" });
    legend($("f-21-dist-legend"), c2);
    const row = (c) => `${IDX[c] || "Probe dependency length"}: ρ ${fmt(S[c].rho_achieved_global, 2)} with the READ-IT global reached (${fmt(S[c].partial_length, 2)} controlling for length), ${fmt(S[c].within_original_vs_achieved, 2)} within original`;
    $("f-21-note").textContent = ["ent", "dist", ...(S.probe_len ? ["probe_len"] : [])].map(row).join("; ") + ".";
    const Q = A.step_2_2, lv5 = [0, 10, 20, 30, 40];
    for (const kind of ["originals", "fixed"]) {
      const ser = ["ent", "dist", "long"].map((c, k) => ({ name: IDX[c], color: palette[["series-1", "series-2", "series-4"][k]], values: lv5.map((l) => Q[kind][l][c].relative_to_40) }));
      lineChart($(`f-22-${kind}`), { xs: lv5, series: ser, height: 190, xTitle: "Level the text is read under", yFormat: (v) => `${(100 * v).toFixed(1)}%`,
        zero: false, yLo: -0.05, yHi: 0.05, ref: { name: "level 40", values: lv5.map(() => 0) }, label: `Change of the syntax-head indices relative to level 40, ${kind}` });
      legend($(`f-22-${kind}-legend`), ser);
      const pu = lv5.map((l) => Q[kind][l].probe_uuas);
      $(`f-22-${kind}-note`).textContent = pu[0] != null ? `Probe UUAS at levels 0–40: ${pu.map((x) => fmt(x, 3)).join(", ")}.` : "";
    }
  }

  // ---------- wiring ----------
  function renderAll() {
    readPalette();
    $("meta").textContent = `Built ${D.meta.generated}.`;
    renderA(); renderB(); renderC(); renderD(); renderE(); renderF();
  }
  const THEMES = ["auto", "light", "dark"];
  let theme = "auto";
  function applyTheme(next) {
    theme = THEMES.includes(next) ? next : "auto";
    if (theme === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = theme;
    $("theme-toggle").textContent = `Theme: ${theme}`;
  }
  function init() {
    if (!D) { document.querySelector("main").textContent = "No data loaded."; return; }
    let saved = "auto";
    try { saved = localStorage.getItem("webview-theme") || "auto"; } catch (e) { /* storage unavailable */ }
    applyTheme(saved);
    $("theme-toggle").addEventListener("click", () => {
      applyTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]);
      try { localStorage.setItem("webview-theme", theme); } catch (e) { /* storage unavailable */ }
      renderAll();
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
    const slider = $("layer");
    slider.max = String(geo().layers.length - 1);
    slider.addEventListener("input", (e) => { state.layer = Number(e.target.value); renderA(); });
    segmented("a-model", MODEL_LABEL, state.model, (v) => { state.model = v; renderA(); });
    segmented("a-view", VIEWS, state.view, (v) => { state.view = v; renderA(); });
    segmented("a-proj", PROJ, state.proj, (v) => { state.proj = v; renderA(); });
    segmented("a-variant", VARIANTS, state.variant, (v) => { state.variant = v; renderA(); });
    segmented("e-model", MODEL_LABEL, state.eModel, (v) => { state.eModel = v; renderE(); });
    const pick = $("e-pick");
    D.classes[state.eModel].examples.forEach((ex, i) => pick.append(h("option", { value: i, text: `${i + 1}. ${ex.original.slice(0, 70)}…` })));
    pick.addEventListener("change", (e) => { state.example = Number(e.target.value); renderExample(); });
    renderAll();
    let timer;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  }
  init();
})();
