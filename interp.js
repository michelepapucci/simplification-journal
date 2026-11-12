(() => {
  "use strict";

  const D = window.INTERP_DATA;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const NF = new Intl.NumberFormat("en-US");
  const DIMS = ["base", "lexical", "syntax"];
  const PAIRS = ["base-lexical", "base-syntax", "lexical-syntax"];
  const SCORES = ["readit_base", "readit_lexical", "readit_syntax", "readit_all"];
  const DIM_COLOR = { base: "series-1", lexical: "series-2", syntax: "series-3" };
  const PAIR_COLOR = { "base-lexical": "series-4", "base-syntax": "series-8", "lexical-syntax": "series-5" };
  const cap = (s) => s[0].toUpperCase() + s.slice(1);
  const MODES = { free: "any (no request)", keep: "held at the original’s levels" };
  const LABEL = {
    readit_base: "READ-IT base", readit_lexical: "READ-IT lexical", readit_syntax: "READ-IT syntax", readit_all: "READ-IT global",
    n_tokens: "Tokens", n_sentences: "Sentences", tokens_per_sent: "Tokens per sentence", char_per_tok: "Characters per token",
    in_dict: "Basic-vocabulary share", lexical_density: "Lexical density", upos_dist_PUNCT: "Punctuation share",
    subordinate_proposition_dist: "Share of subordinate clauses", avg_max_depth: "Parse-tree depth", avg_links_len: "Dependency length",
    verbal_head_per_sent: "Verbal heads per sentence", len_ratio: "Length, output / original", kept: "Share of original tokens kept",
    deleted: "Share of original tokens deleted", replaced: "Share of original tokens replaced", inserted: "Share of output tokens inserted",
    copy: "Copies of the original",
  };
  const label = (f) => LABEL[f] || f;
  const state = { mode: "free", example: 0, pair: "base-syntax", cmp: "matched", category: "", search: "", sigOnly: true, allRows: false, sort: { key: "max", dir: "desc" } };

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
  const dimColor = (d) => palette[DIM_COLOR[d.replace("readit_", "")]] || palette["text-muted"];
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
  function lineChart(container, { xs, series, height = 210, xTitle, yFormat = (v) => fmt(v, 1), yLo, yHi, ref, label: aria, xFormat = (v) => v }) {
    const width = Math.max(220, Math.floor(container.clientWidth || 300));
    const m = { top: 10, right: 12, bottom: 38, left: 46 };
    const vals = series.flatMap((se) => se.values.filter((v) => v != null));
    if (ref) vals.push(...ref.values.filter((v) => v != null));
    let lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
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
  const dimLegend = () => DIMS.map((d) => ({ name: `${cap(d)} token moved`, color: dimColor(d) }));

  // ---------- header and findings ----------
  function renderHeader() {
    $("meta").textContent = `${D.label}: ${NF.format(D.n_originals)} test originals, ${NF.format(D.n_requests)} greedy generations ` +
      `(${NF.format(D.n_distinct_outputs)} distinct texts). Built ${D.generated}.`;
    const ul = clear($("findings"));
    for (const f of D.findings || []) ul.append(h("li", { text: f }));
    $("mode-summary").textContent = state.mode === "free"
      ? "Others any: only the lowered token is set, as in the main evaluation. Every result below is shown separately for base, lexical and syntax lowered."
      : "Others held: the two other tokens stay at the original’s own levels (a held lexical token is outside v5’s training data). Results are per lowered token.";
  }

  // ---------- Q1: example ----------
  function highlight(text, original) {
    const words = new Set((original.toLowerCase().match(/[\p{L}\p{N}]+/gu) || []));
    const out = [];
    for (const part of text.split(/([\p{L}\p{N}]+)/u)) {
      if (!part) continue;
      if (/^[\p{L}\p{N}]+$/u.test(part) && !words.has(part.toLowerCase())) out.push(h("mark", { class: "new", text: part }));
      else out.push(part);
    }
    return out;
  }
  function renderExample() {
    const ex = D.examples[state.example];
    const o = $("example-original");
    clear(o).append(h("span", { class: "tag", text: `Original · base ${ex.src.base} · lexical ${ex.src.lexical} · syntax ${ex.src.syntax}` }), ex.original);
    const table = clear($("example"));
    const rows = ex.outputs.filter((r) => r.mode === state.mode);
    const maxLevel = Math.max(...DIMS.map((d) => ex.src[d])) - 10;
    table.append(h("thead", {}, h("tr", {}, h("th", { text: "Level" }), DIMS.map((d) => h("th", { text: `${cap(d)} token` })))));
    const body = h("tbody");
    for (let lv = maxLevel; lv >= 0; lv -= 10) {
      const tr = h("tr", {}, h("td", { class: "level", text: lv }));
      for (const d of DIMS) {
        const r = rows.find((x) => x.dim === d && x.level === lv);
        if (!r) { tr.append(h("td", { class: "same", text: lv >= ex.src[d] ? "at or above the original" : "–" })); continue; }
        const above = rows.find((x) => x.dim === d && x.level === lv + 10);
        const same = above && above.text === r.text;
        const sc = r.scores.slice(0, 3).map((v) => (v == null ? "–" : v.toFixed(0))).join(" · ");
        tr.append(h("td", { class: same ? "same" : null },
          same ? "same as above" : highlight(r.text, ex.original), h("span", { class: "score", text: sc })));
      }
      body.append(tr);
    }
    table.append(body);
  }

  // ---------- Q1: edit operations ----------
  const EDIT_PANELS = ["kept", "deleted", "replaced", "inserted", "n_sentences", "n_tokens", "char_per_tok", "in_dict"];
  function renderEdits() {
    const q = D.q1[state.mode];
    $("edits-sub").textContent = "Within-original change from the output one level below the original, as the moved token asks for a simpler output. " +
      "Each point averages the originals that can be lowered that far (at least 100). The note gives the mean one level below the original.";
    legend($("edits-legend"), dimLegend());
    const box = clear($("edits"));
    for (const f of EDIT_PANELS) {
      const panel = h("div", { class: "multiple" });
      box.append(panel);
      const at1 = DIMS.map((d) => `${d} ${fmt(q.curves[d].at1[f], f.startsWith("n_") || f === "char_per_tok" ? 1 : 2)}`).join(" · ");
      panel.append(h("h4", { text: label(f) }), h("p", { class: "note", text: `one level below: ${at1}` }));
      const xs = [...new Set(DIMS.flatMap((d) => q.curves[d].drop))].sort((a, b) => a - b);
      const series = DIMS.map((d) => ({ name: `${cap(d)} token`, color: dimColor(d), values: xs.map((k) => { const i = q.curves[d].drop.indexOf(k); return i < 0 ? null : q.curves[d][f][i]; }) }));
      const chart = h("div", { class: "chart" });
      panel.append(chart);
      const isShare = !f.startsWith("n_") && f !== "char_per_tok";
      lineChart(chart, { xs, series, height: 180, xTitle: "Levels below the original", label: `${label(f)}: change by request strength`,
        yFormat: isShare ? (v, st) => signed(v * 100, st * 100 < 1 ? 1 : 0) + (v === 0 ? "" : " pt") : (v, st) => signed(v, st < 0.1 ? 2 : 1) });
    }
  }

  // ---------- Q1: same or different edits ----------
  function renderSame() {
    const q = D.q1[state.mode], t = clear($("same"));
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "Test" }), h("th", { text: "Result" }), h("th", { text: "Reading" }))));
    const b = h("tbody");
    const row = (test, result, reading) => b.append(h("tr", {}, h("td", { class: "wrap-text", text: test }), h("td", { class: "wrap-text", text: result }), h("td", { class: "reading", text: reading })));
    row("One level apart, same token: the very same text",
      DIMS.map((d) => `${d} ${pct(q.identical.adjacent[d])}`).join(" · "),
      "How often moving a token by one level changes nothing at all.");
    row("Two tokens at the same distance from the original: the very same text (1 / 2 / 3 levels below)",
      PAIRS.map((p) => `${p.replace("-", "–")} ${[1, 2, 3].map((k) => pct(q.identical.across[p][k])).join(" / ")}`).join("; "),
      "If the tokens were interchangeable, these would be high.");
    row("Profile similarity: Spearman, across features, of the effects in the table below",
      PAIRS.map((p) => `${p.replace("-", "–")} ${fmt(q.profile_rho[p])}`).join(" · "),
      "Near 1: the same edits, at different strengths. Lower: different edits.");
    const acc = q.classifier.accuracy;
    row(`Which token was moved, told from the output (1–3 levels below, 3 classes, chance ${pct(q.classifier.chance)})`,
      Object.entries(acc).map(([k, v]) => `${k}: ${pct(v)}`).join(" · "),
      "From the overall READ-IT change alone versus the linguistic profile: the gap is what the tokens change beyond the overall score.");
    const nc = q.classifier_noncopy;
    row("The same, without copies of the original",
      Object.entries(nc.accuracy).map(([k, v]) => `${k}: ${pct(v)}`).join(" · "),
      `${NF.format(nc.n)} outputs. A copy is the same text whichever token produced it, so it cannot be told apart.`);
    t.append(b);
  }

  // ---------- Q1: feature table ----------
  function rbar(r) {
    const w = 64, svg = s("svg", { width: w, height: 10, "aria-hidden": "true" });
    svg.append(s("line", { x1: w / 2, x2: w / 2, y1: 0, y2: 10, stroke: palette["text-muted"], "stroke-width": 1 }));
    if (r != null) {
      const len = Math.abs(r) * (w / 2);
      svg.append(s("rect", { x: r >= 0 ? w / 2 : w / 2 - len, y: 1, width: Math.max(1, len), height: 8, rx: 2, fill: divergingColor(r) }));
    }
    return svg;
  }
  /** Index of the largest |r| among the three tokens (null when none is defined). */
  const top = (rs) => { let best = null; rs.forEach((v, k) => { if (v != null && (best == null || Math.abs(v) > Math.abs(rs[best]))) best = k; }); return best; };
  function renderFeatures() {
    const q = D.q1[state.mode];
    $("features-sub").textContent = "Paired comparison, same original: the output three levels below the original against the output one level below " +
      "(rank-biserial r of the differences; Wilcoxon, Benjamini–Hochberg q), separately for each lowered token. Positive: the feature grows as the token asks for a simpler output. " +
      "In each row, the token with the largest effect is in bold; grey: q ≥ 0.01.";
    const lg = clear($("features-legend"));
    lg.append(h("span", { text: "r −1" }), h("span", { class: "bar", style: `background:linear-gradient(90deg, ${palette["div-neg"]}, ${palette["div-mid"]}, ${palette["div-pos"]})` }),
      h("span", { text: "+1 · blue: decreases, red: increases" }));
    const rows = D.features.map((f, i) => ({
      f, i, cat: D.categories[i],
      r: DIMS.map((d) => q.contrast[d].r[i]), q: DIMS.map((d) => q.contrast[d].q[i]), md: DIMS.map((d) => q.contrast[d].mean_diff[i]),
    })).filter((r) => (!state.category || r.cat === state.category)
      && (!state.search || r.f.toLowerCase().includes(state.search.toLowerCase()) || label(r.f).toLowerCase().includes(state.search.toLowerCase()))
      && (!state.sigOnly || r.q.some((v) => v != null && v < 0.01)));
    const keyOf = (r) => (state.sort.key === "max" ? Math.max(...r.r.map((v) => Math.abs(v ?? 0))) : state.sort.key === "name" ? r.f : (r.r[DIMS.indexOf(state.sort.key)] ?? -9));
    rows.sort((a, b) => { const A = keyOf(a), B = keyOf(b); const c = typeof A === "string" ? A.localeCompare(B) : A - B; return state.sort.dir === "asc" ? c : -c; });
    const t = clear($("features"));
    const head = (key, text, num) => {
      const th = h("th", { class: num ? "num" : null, "aria-sort": state.sort.key === key ? (state.sort.dir === "asc" ? "ascending" : "descending") : null });
      th.append(h("button", { type: "button", text, onclick: () => { state.sort = { key, dir: state.sort.key === key && state.sort.dir === "desc" ? "asc" : "desc" }; renderFeatures(); } }));
      return th;
    };
    t.append(h("thead", {}, h("tr", {}, head("name", "Feature"), DIMS.map((d) => head(d, `${cap(d)} lowered: r`, true)), head("max", "Largest |r|", true))));
    const body = h("tbody");
    const shown = state.allRows || state.search ? rows : rows.slice(0, 30);
    for (const r of shown) {
      body.append(h("tr", {},
        h("td", { class: "feature" }, LABEL[r.f] ? `${LABEL[r.f]} ` : null, LABEL[r.f] ? h("span", { class: "cat", text: r.f }) : r.f, h("span", { class: "cat", text: r.cat })),
        DIMS.map((d, k) => h("td", { class: `num${r.q[k] == null || r.q[k] >= 0.01 ? " muted" : ""}${k === top(r.r) ? " top" : ""}`, title: `mean change ${signed(r.md[k], 3)}, q ${fmtQ(r.q[k])}` },
          h("span", { class: "rbar" }, signed(r.r[k]), rbar(r.r[k])), h("span", { class: "nd", text: `Δ ${signed(r.md[k], 3)}` }))),
        h("td", { class: "num", text: fmt(Math.max(...r.r.map((v) => Math.abs(v ?? 0)))) })));
    }
    t.append(body);
    $("row-count").textContent = `${shown.length} shown · ${rows.length} of ${D.features.length} features match`;
  }

  // ---------- Q1: matched ----------
  function renderMatched() {
    const q = D.q1[state.mode], [a, b] = state.pair.split("-");
    const m = q[state.cmp][state.pair];
    $("matched-sub").textContent = state.cmp === "matched"
      ? `Same sentence: the ${a}-lowered output and the ${b}-lowered output (any levels, different texts, neither a copy) whose overall READ-IT scores are closest, ` +
        `kept if within 2 points: ${NF.format(m.n)} sentences, mean gap ${fmt(m.mean_gap, 2)} points. The amount of simplification is matched, so what still differs ` +
        `is what each token edits. Positive r: higher when ${a} is lowered. Features with q < 0.01, largest |r| first.`
      : `Same sentence, ${a} lowered and ${b} lowered by the same number of levels (1, 2 and 3 below the original, averaged): ${NF.format(m.n)} sentences; ` +
        `the two outputs are the very same text ${pct(m.same_text)} of the time. The same distance is not the same strength (see experiment 4), so these differences ` +
        `mix how much and what. Positive r: higher when ${a} is lowered. Features with q < 0.01, largest |r| first.`;
    const rows = D.features.map((f, i) => ({ f, i, r: m.r[i], q: m.q[i], md: m.mean_diff[i], cat: D.categories[i] }))
      .filter((r) => r.r != null && r.q != null && r.q < 0.01).sort((x, y) => Math.abs(y.r) - Math.abs(x.r)).slice(0, 30);
    const t = clear($("matched"));
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "Feature" }), h("th", { class: "num", text: "r" }),
      h("th", { class: "num", text: `Mean, ${a} lowered − ${b} lowered` }), h("th", { class: "num", text: "q" }))));
    const body = h("tbody");
    for (const r of rows) {
      body.append(h("tr", {},
        h("td", { class: "feature" }, LABEL[r.f] ? `${LABEL[r.f]} ` : null, LABEL[r.f] ? h("span", { class: "cat", text: r.f }) : r.f, h("span", { class: "cat", text: r.cat })),
        h("td", { class: "num" }, h("span", { class: "rbar" }, signed(r.r), rbar(r.r))),
        h("td", { class: "num", text: signed(r.md, 3) }), h("td", { class: "num", text: fmtQ(r.q) })));
    }
    if (!rows.length) body.append(h("tr", {}, h("td", { colspan: 4, text: "No feature differs at q < 0.01." })));
    t.append(body);
  }

  // ---------- Q2: matrix ----------
  function renderMatrix() {
    const t = clear($("matrix"));
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "Moved token" }), SCORES.map((e) => h("th", { text: label(e).replace("READ-IT ", "Achieved ") })))));
    const body = h("tbody");
    const cell = (v, e, ci) => {
      const tint = v == null ? palette.surface : mix(palette.surface, e === "readit_all" ? palette["text-muted"] : dimColor(e), Math.min(0.5, Math.abs(v) * 0.5));
      return h("td", { class: "v", style: `background:${tint}` }, h("strong", { text: fmt(v) }), ci ? h("span", { class: "ci", text: `[${fmt(ci[0])}, ${fmt(ci[1])}]` }) : null);
    };
    for (const mode of ["free", "keep"]) {
      DIMS.forEach((d, k) => {
        const sl = D.q2[mode].slopes[d];
        body.append(h("tr", { class: k === 0 ? "sep" : null }, h("td", {}, h("strong", { text: `${cap(d)} lowered` }), ` · ${mode === "free" ? "others any" : "others held"}`),
          SCORES.map((e) => cell(sl[e].slope, e, [sl[e].lo, sl[e].hi]))));
      });
    }
    if (D.q2.data) {
      DIMS.forEach((d, k) => body.append(h("tr", { class: `ref${k === 0 ? " sep" : ""}` }, h("td", { text: `${cap(d)} · training data` }), SCORES.map((e) => cell(D.q2.data[d][e], e)))));
    }
    t.append(body);
  }

  // ---------- Q2: dose-response ----------
  function renderDose() {
    const q = D.q1[state.mode];
    $("dose-sub").textContent = "Change in each achieved READ-IT score, within the sentence, from the output one level below the original, as one token moves down " +
      "(at least 100 sentences per point). The dotted line is what that token asks for: 10 points per level.";
    legend($("dose-legend"), [...DIMS.map((d) => ({ name: `Achieved ${d}`, color: dimColor(d) })), { name: "Requested change", color: palette["text-muted"], dashed: true }]);
    const box = clear($("dose"));
    for (const d of DIMS) {
      const c = q.curves[d], panel = h("div", { class: "multiple" });
      box.append(panel);
      panel.append(h("h4", { text: `${cap(d)} token moved` }),
        h("p", { class: "note", text: `slopes: ${SCORES.slice(0, 3).map((e) => `${e.replace("readit_", "")} ${fmt(D.q2[state.mode].slopes[d][e].slope)}`).join(" · ")}` }));
      const chart = h("div", { class: "chart" });
      panel.append(chart);
      lineChart(chart, { xs: c.drop, height: 230, xTitle: "Levels below the original", label: `Achieved scores as the ${d} token moves`,
        series: SCORES.slice(0, 3).map((e) => ({ name: `Achieved ${e.replace("readit_", "")}`, color: dimColor(e), values: c[e] })),
        ref: { name: "Requested change", values: c.drop.map((k) => -10 * (k - 1)) }, yFormat: (v) => signed(v, 0) });
    }
  }

  // ---------- Q2: directions ----------
  function renderDirections() {
    const e = D.q2.embeddings, a = D.q2[state.mode].activations;
    $("dirs-sub").textContent = "Each token’s level direction: in the trained input embeddings (the regression of its ten level embeddings on the level), and in the residual " +
      "stream at the last prompt token, where generation starts (the within-original regression of the state on the requested level, at every layer). " +
      "Right: each token’s requested level read along a direction estimated on the other half of the originals (correlation, within original). " +
      "Solid: along the token’s own direction; dashed: along another token’s.";
    const t = clear($("embed"));
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "Input embeddings" }), PAIRS.map((p) => h("th", { class: "num", text: p.replace("-", "–") })), h("th", { class: "num", text: "Null, 95%" }))));
    t.append(h("tbody", {}, h("tr", {}, h("td", { text: "Cosine between level directions" }), PAIRS.map((p) => h("td", { class: "num", text: fmt(e.cos[p]) })), h("td", { class: "num", text: `±${fmt(e.null95)}` })),
      h("tr", {}, h("td", { text: "First principal component vs level: Spearman (variance share)" }), DIMS.map((d) => h("td", { class: "num", text: `${d} ${fmt(e.pc1[d].rho)} (${pct(e.pc1[d].share)})` })), h("td", {}))));
    const box = clear($("dirs"));
    const p1 = h("div", { class: "multiple" }), p2 = h("div", { class: "multiple" });
    box.append(p1, p2);
    p1.append(h("h4", { text: "Cosine between level directions, by layer" }), h("div", { class: "legend", id: "dirs-legend-1" }), h("div", { class: "chart", id: "dirs-chart-1" }));
    p2.append(h("h4", { text: "Reading each requested level along each direction" }), h("div", { class: "legend", id: "dirs-legend-2" }), h("div", { class: "chart", id: "dirs-chart-2" }));
    const s1 = [...PAIRS.map((p) => ({ name: p.replace("-", "–"), color: palette[PAIR_COLOR[p]], values: a.cos[p] }))];
    legend($("dirs-legend-1"), s1);
    lineChart($("dirs-chart-1"), { xs: a.layer, series: s1, height: 240, xTitle: "Layer (0 = embeddings)", yLo: -0.2, yHi: 1, label: "Cosine between level directions by layer", yFormat: (v) => fmt(v, 1) });
    const s2 = [
      ...DIMS.map((d) => ({ name: `${cap(d)} level, ${d} direction`, color: dimColor(d), values: a.readout[`${d}>${d}`] })),
      { name: "Syntax level, base direction", color: dimColor("syntax"), values: a.readout["base>syntax"], dashed: true },
      { name: "Base level, syntax direction", color: dimColor("base"), values: a.readout["syntax>base"], dashed: true },
      { name: "Lexical level, base direction", color: dimColor("lexical"), values: a.readout["base>lexical"], dashed: true },
    ];
    legend($("dirs-legend-2"), s2);
    lineChart($("dirs-chart-2"), { xs: a.layer, series: s2, height: 240, xTitle: "Layer (0 = embeddings)", yLo: -0.2, yHi: 1, label: "Readout of requested levels by layer", yFormat: (v) => fmt(v, 1) });
  }

  function renderMethods() {
    $("footer").textContent = D.footer || "";
  }

  // ---------- controls ----------
  function segmented(id, options, prop, onChange) {
    const box = clear($(id));
    for (const [v, text] of Object.entries(options)) {
      box.append(h("button", { type: "button", role: "radio", "data-value": v, "aria-checked": String(state[prop] === v), text,
        onclick: () => { state[prop] = v; box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === v))); onChange(); } }));
    }
  }
  function renderAll() {
    readPalette();
    renderHeader(); renderExample(); renderEdits(); renderSame(); renderFeatures(); renderMatched(); renderMatrix(); renderDose(); renderDirections(); renderMethods();
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
    segmented("mode", MODES, "mode", renderAll);
    segmented("pair", Object.fromEntries(PAIRS.map((p) => [p, p.replace("-", " vs ")])), "pair", renderMatched);
    segmented("cmp", { matched: "same overall READ-IT", same_distance: "same distance" }, "cmp", renderMatched);
    const pick = $("example-pick");
    D.examples.forEach((ex, i) => pick.append(h("option", { value: i, text: `${i + 1}. ${ex.original.slice(0, 60)}…` })));
    pick.addEventListener("change", (e) => { state.example = Number(e.target.value); renderExample(); });
    const cat = $("category");
    cat.append(h("option", { value: "", text: "all" }), ...[...new Set(D.categories)].map((c) => h("option", { value: c, text: c })));
    cat.addEventListener("change", (e) => { state.category = e.target.value; renderFeatures(); });
    $("search").addEventListener("input", (e) => { state.search = e.target.value; renderFeatures(); });
    $("sig-only").addEventListener("change", (e) => { state.sigOnly = e.target.checked; renderFeatures(); });
    $("all-rows").addEventListener("change", (e) => { state.allRows = e.target.checked; renderFeatures(); });
    renderAll();
    let timer;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  }
  init();
})();
