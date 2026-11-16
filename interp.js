(() => {
  "use strict";

  // data_interp.js: window.INTERP_G1, built by scripts/build_interp_webview.py from experiments/results/interp_g1.json.
  const D = window.INTERP_G1;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const NF = new Intl.NumberFormat("en-US");
  const state = { model: "g1_guided", view: "in_context", layer: 12, pushModel: "g1_guided" };

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
  const TRAINED = 9;                                   // levels 0-80; 90 is never a training target
  const VIEWS = { in_context: "In context", alone: "Alone" };
  const MODEL_LABEL = Object.fromEntries((D ? D.meta.models : []).map((m) => [m.id, m.label]));
  const tokName = (i) => (i === 10 ? "no level" : i === 9 ? "90 (never trained)" : `level ${i * 10}`);
  const levelColor = (i) => (i < TRAINED ? mix(palette.surface, palette["series-1"], 0.28 + 0.72 * (i / (TRAINED - 1))) : palette["text-muted"]);
  const geo = () => D.models[state.model].geometry;
  const layerData = () => geo().layers[state.layer][state.view];

  function segmented(id, options, value, onPick) {
    const box = clear($(id));
    for (const [v, text] of Object.entries(options)) {
      box.append(h("button", { type: "button", role: "radio", "aria-checked": String(v === value), text,
        onclick: () => { onPick(v); box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.textContent === text))); } }));
    }
  }

  // ---------- A: the level map at one layer ----------
  function levelMap(container) {
    clear(container);
    const L = layerData(), pts = L.coords, cloud = state.view === "in_context" ? L.cloud : null;
    const width = Math.max(260, Math.floor(container.clientWidth || 480)), height = Math.min(420, Math.max(260, Math.round(width * 0.75)));
    const m = { top: 14, right: 18, bottom: 34, left: 18 };
    if (pts.every((p) => p[0] === 0 && p[1] === 0)) {
      container.append(h("p", { class: "sub", text: "At this layer the state where generation starts has not yet read the control token: all tokens give the same state." }));
      return;
    }
    const all = [...pts, ...(cloud ? cloud.flat() : [])].filter((p) => p[0] != null);
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)], [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.08 || 1;
    const x = (v) => m.left + ((v - x0 + px) / (x1 - x0 + 2 * px)) * (width - m.left - m.right);
    const y = (v) => height - m.bottom - ((v - y0 + py) / (y1 - y0 + 2 * py)) * (height - m.top - m.bottom);
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", tabindex: 0,
      "aria-label": `The 11 control tokens at layer ${state.layer}, projected on the two main directions of the trained levels` });
    svg.append(s("line", { class: "baseline", x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }));
    svg.append(s("text", { class: "axis-title", x: m.left + (width - m.left - m.right) / 2, y: height - 8, "text-anchor": "middle",
      text: `Main direction (PC1, ${pct(L.pc1_share)} of the spread between level means)` }));
    if (cloud) {
      cloud.forEach((row) => row.forEach((p, j) => {
        if (p[0] == null || j >= 10) return;
        svg.append(s("circle", { cx: x(p[0]), cy: y(p[1]), r: 2, fill: levelColor(j), opacity: 0.35 }));
      }));
    }
    svg.append(s("path", { d: `M${pts.slice(0, TRAINED).map((p) => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("L")}`,
      fill: "none", stroke: palette["text-secondary"], "stroke-width": 1.5, "stroke-linejoin": "round" }));
    pts.forEach((p, i) => {
      const cx = x(p[0]), cy = y(p[1]);
      let mark;
      if (i === 10) mark = s("rect", { x: cx - 6, y: cy - 6, width: 12, height: 12, transform: `rotate(45 ${cx} ${cy})`, fill: palette["series-4"], stroke: palette.surface, "stroke-width": 2 });
      else if (i === 9) mark = s("circle", { cx, cy, r: 6, fill: palette.surface, stroke: palette["text-muted"], "stroke-width": 2 });
      else mark = s("circle", { cx, cy, r: 7, fill: levelColor(i), stroke: palette.surface, "stroke-width": 2 });
      svg.append(mark);
      svg.append(s("text", { class: "axis-text", x: cx + 9, y: i === 10 ? cy + 18 : cy - 8, text: i === 10 ? "no level" : String(i * 10) }));
      const hit = s("circle", { cx, cy, r: 12, fill: "transparent" });
      hit.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, (t) => {
        t.append(h("div", { class: "tt-title", text: `${tokName(i)} · layer ${state.layer}` }));
        t.append(h("div", { class: "tt-note", text: `PC1 ${fmt(p[0], 2)}, PC2 ${fmt(p[1], 2)}` }));
        if (i >= 9) t.append(h("div", { class: "tt-note", text: `sits where level ${fmt(L.equivalent_level[i === 9 ? "90" : "any"], 0)} would be on the main direction` }));
      }));
      hit.addEventListener("pointerleave", hideTip);
      svg.append(hit);
    });
    container.append(svg);
  }

  function cosineMatrix(container) {
    clear(container);
    const C = layerData().cosine;
    if (!C || C[0][0] == null) return;
    const names = D.meta.tokens.map((t) => (t === "any" ? "no level" : t));
    const t = h("table", { class: "data cosine" });
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "" }), names.map((n) => h("th", { class: "num", text: n })))));
    const body = h("tbody");
    C.forEach((row, i) => body.append(h("tr", {}, h("th", { scope: "row", text: names[i] }), row.map((v, j) => {
      const bg = divergingColor(-v);
      return h("td", { class: "num", style: `background:${bg};color:${Math.abs(v) > 0.6 ? palette.surface : palette["text-primary"]}`,
        title: `${names[i]} vs ${names[j]}: ${fmt(v, 2)}`, text: fmt(v, 1) });
    }))));
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
      label: "Ordering and hierarchy measures of the level tokens at every layer" });
    return series;
  }

  function spacingChart(container) {
    clear(container);
    const sp = layerData().spacing;
    if (!sp || sp[0] == null) return [];
    const beh = D.behaviour[state.model] || {};
    const gaps = [0, 10, 20, 30, 40, 50, 60].map((lv) => (beh[lv + 10] != null && beh[lv] != null ? beh[lv + 10] - beh[lv] : null));
    const meanGap = gaps.filter((v) => v != null).reduce((a, b) => a + b, 0) / gaps.filter((v) => v != null).length;
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
    const L = layerData();
    $("a-stats").textContent = L.rho_pc1 == null ? "" :
      `Layer ${state.layer}, ${VIEWS[state.view].toLowerCase()}: order ρ ${fmt(L.rho_pc1, 2)}, in-between ${pct(L.betweenness)}, neighbour merges ${pct(L.contiguous_merges)}` +
      (L.readout_r2 != null ? `, level read back R² ${fmt(L.readout_r2, 3)}` : "") + `. “No level” sits at level ${fmt(L.equivalent_level.any, 0)}, the untrained level 90 at ${fmt(L.equivalent_level["90"], 0)}.`;
    levelMap($("a-map"));
    cosineMatrix($("a-cosine"));
    legend($("a-over-legend"), overLayers($("a-over")));
    legend($("a-spacing-legend"), spacingChart($("a-spacing")));
    const e = geo().embeddings.input;
    $("a-emb").textContent = `Input embeddings of the tokens (${MODEL_LABEL[state.model]}): order ρ ${fmt(e.rho_pc1, 2)}, in-between ${pct(e.betweenness)}, ` +
      `neighbour merges ${pct(e.contiguous_merges)}. The output layer never predicts these tokens, so its rows keep their initial values.`;
  }

  // ---------- B: readout ----------
  function renderB() {
    const box = $("b-chart");
    const xs = D.models[D.meta.models[0].id].readout.state_by_layer.map((_, i) => i);
    const series = D.meta.models.map((m, k) => ({ name: `${m.label}: hidden state at that layer`, color: palette[k ? "series-2" : "series-1"],
      values: D.models[m.id].readout.state_by_layer }));
    const r0 = D.models[D.meta.models[0].id].readout;
    series.push({ name: "The request alone", color: palette["text-secondary"], values: xs.map(() => r0.request), dashed: true });
    series.push({ name: "The request and the original’s READ-IT and length", color: palette["series-3"], values: xs.map(() => r0.request_and_original), dashed: true });
    lineChart(box, { xs, series, height: 230, xTitle: "Layer", yFormat: (v) => fmt(v, 2), yLo: 0, yHi: 0.35, label: "How well each predictor explains the READ-IT the output reaches" });
    legend($("b-legend"), series);
  }

  // ---------- C: no level ----------
  function renderC() {
    const n = D.no_level;
    if (!n) return;
    const box = clear($("c-bars"));
    const levels = Object.keys(n.level_shares).map(Number);
    const max = Math.max(...Object.values(n.level_shares));
    const t = h("table", { class: "data bars" });
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "READ-IT level reached" }), h("th", { class: "num", text: "Outputs" }), h("th", { text: "" }))));
    const body = h("tbody");
    for (const lv of levels) {
      const v = n.level_shares[lv];
      body.append(h("tr", {}, h("td", { text: `${lv}–${lv + 10}` }), h("td", { class: "num", text: pct(v) }),
        h("td", { class: "barcell" }, h("span", { class: "bar", style: `width:${(100 * v) / max}%;background:${palette["series-4"]}` }))));
    }
    t.append(body);
    box.append(t);
    const ls = D.models.g1_guided.geometry.layers;
    const xs = ls.map((_, i) => i);
    const series = [
      { name: "“No level” (g1 · guided)", color: palette["series-4"], values: ls.map((l) => l.in_context.equivalent_level.any) },
      { name: "Level 90, never trained (g1 · guided)", color: palette["text-muted"], values: ls.map((l) => l.in_context.equivalent_level["90"]), dashed: true },
    ];
    lineChart($("c-chart"), { xs, series, height: 210, xTitle: "Layer", yFormat: (v) => fmt(v, 0), yLo: 0, yHi: 80,
      label: "Where the no-level token and the untrained level sit on the level direction, by layer" });
    legend($("c-legend"), series);
  }

  // ---------- D: vocabulary ----------
  const COND = { g1: ["g1", "series-1", false], g1_guided: ["g1 · guided, w = 1", "series-3", false], g1_guided_w3: ["g1 · guided, w = 3", "series-3", true] };
  function renderD() {
    const V = D.vocabulary, levels = [0, 10, 20, 30, 40, 50, 60, 70];
    const byLevel = (cond, key) => levels.map((lv) => { const b = V.generated[cond] && V.generated[cond].by_level && V.generated[cond].by_level[lv]; return b ? b[key] : null; });
    const mk = (key) => Object.entries(COND).filter(([c]) => V.generated[c]).map(([c, [name, col, dashed]]) => ({ name, color: palette[col], dashed, values: byLevel(c, key) }));
    const pf = (v, step) => `${(100 * v).toFixed(step < 0.01 ? 1 : 0)}%`;
    const nl = V.generated.g1_guided_null;
    const s1 = mk("basic_share");
    lineChart($("d-dist"), { xs: levels, series: s1, height: 220, xTitle: "Requested level", yFormat: pf, zero: false, yLo: 0.85, yHi: 0.95,
      ref: nl ? { name: "“No level”", values: levels.map(() => nl.all.basic_share) } : null,
      label: "Share of the next-token probability on De Mauro basic vocabulary, by requested level" });
    legend($("d-dist-legend"), [...s1, ...(nl ? [{ name: "“No level” (g1 · guided)", color: palette["text-muted"], dashed: true }] : [])]);
    const s2 = mk("basic_share_text");
    const refs = levels.map((lv) => (V.references_text_by_level[lv] ? V.references_text_by_level[lv].basic_share_text : null));
    s2.push({ name: "IMPaCTS references at that level", color: palette["series-2"], values: refs });
    lineChart($("d-text"), { xs: levels, series: s2, height: 220, xTitle: "Requested level", yFormat: pf, zero: false, yLo: 0.85, yHi: 0.95,
      ref: { name: "The originals", values: levels.map(() => V.originals_text.basic_share_text) },
      label: "Share of the output's words in De Mauro basic vocabulary, by requested level" });
    legend($("d-text-legend"), [...s2, { name: "The originals", color: palette["text-muted"], dashed: true }]);
    const toks = D.meta.tokens.filter((t) => t !== "any");
    const s3 = D.meta.models.filter((m) => V.fixed[m.id]).map((m, k) => ({ name: m.label, color: palette[k ? "series-1" : "series-3"],
      values: toks.map((t) => V.fixed[m.id].by_token[t] ? V.fixed[m.id].by_token[t].basic_share : null) }));
    const anyG = V.fixed.g1_guided && V.fixed.g1_guided.by_token.any;
    lineChart($("d-fixed"), { xs: toks.map(Number), series: s3, height: 220, xTitle: "Control token (level)", yFormat: pf, zero: false, yLo: 0.85, yHi: 0.95,
      ref: anyG ? { name: "“No level” (g1 · guided)", values: toks.map(() => anyG.basic_share) } : null,
      label: "Share of the next-token probability on basic vocabulary, the same text read under each token" });
    legend($("d-fixed-legend"), [...s3, ...(anyG ? [{ name: "“No level” (g1 · guided)", color: palette["text-muted"], dashed: true }] : [])]);
    renderPushed();
  }
  function renderPushed() {
    const box = clear($("d-pushed"));
    const P = D.vocabulary.pushed[state.pushModel];
    if (!P) return;
    const ref = state.pushModel === "g1_guided" ? "“no level”" : "level 40";
    const cls = { FO: "fondamentale", AU: "alto uso", AD: "alta disponibilità", out: "outside NVdB", "": "" };
    for (const lv of ["0", "70"]) {
      const t = h("table", { class: "data pushed" });
      t.append(h("thead", {}, h("tr", {}, h("th", { text: `Up at level ${lv}` }), h("th", { class: "num", text: "×" }), h("th", { text: `Down at level ${lv}` }), h("th", { class: "num", text: "×" }))));
      const body = h("tbody");
      for (let i = 0; i < 12; i++) {
        const u = P[lv].up[i], d = P[lv].down[i];
        const item = (x) => h("td", {}, h("code", { text: JSON.stringify(x.item).slice(1, -1) }), x.class ? h("span", { class: "cls", text: cls[x.class] }) : null);
        body.append(h("tr", {}, item(u), h("td", { class: "num", text: `${Math.exp(u.delta_logp).toFixed(1)}` }), item(d), h("td", { class: "num", text: `${Math.exp(d.delta_logp).toFixed(2)}` })));
      }
      t.append(body);
      const bc = P[lv].by_class;
      box.append(h("div", { class: "multiple" }, h("h4", { text: `Level ${lv} against ${ref}` }),
        h("p", { class: "note", text: `Mean change by class (×): fondamentale ${Math.exp(bc.FO).toFixed(2)}, alto uso ${Math.exp(bc.AU).toFixed(2)}, alta disponibilità ${Math.exp(bc.AD).toFixed(2)}, outside NVdB ${Math.exp(bc.out).toFixed(2)}` }),
        h("div", { class: "scroll-x" }, t)));
    }
  }

  // ---------- wiring ----------
  function renderAll() {
    readPalette();
    $("meta").textContent = `Built ${D.meta.generated}.`;
    renderA(); renderB(); renderC(); renderD();
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
    const nLayers = geo().layers.length;
    const slider = $("layer");
    slider.max = String(nLayers - 1);
    slider.addEventListener("input", (e) => { state.layer = Number(e.target.value); renderA(); });
    segmented("a-model", MODEL_LABEL, state.model, (v) => { state.model = v; renderA(); });
    segmented("a-view", VIEWS, state.view, (v) => { state.view = v; renderA(); });
    segmented("d-model", MODEL_LABEL, state.pushModel, (v) => { state.pushModel = v; renderPushed(); });
    renderAll();
    let timer;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  }
  init();
})();
