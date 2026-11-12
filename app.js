(() => {
  "use strict";

  const ALL = window.WEBVIEW_DATASETS || {};
  const ORDER = ["clicit2025", "v0", "v1", "v2", "v3", "v4", "v5", "v6"].filter((k) => ALL[k]);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const NF = new Intl.NumberFormat("en-US");
  const DIMS = { all: "Global", base: "Base", lexical: "Lexical", syntax: "Syntax" };
  const COMPARE_FEATURES = ["n_tokens", "char_per_tok", "in_dict", "lexical_density", "avg_max_depth", "verbal_head_per_sent", "subordinate_proposition_dist"];
  let D, SPLITS, STRATEGIES, FEATURES, MIN_PAIRS;
  function useDataset(id) {
    D = ALL[id];
    SPLITS = D.meta.splits;
    STRATEGIES = D.meta.strategies;
    FEATURES = D.features.map((f) => f.name);
    MIN_PAIRS = D.meta.min_pairs;
  }

  const state = {
    data: ORDER[ORDER.length - 1], split: "train", strategy: "base", level: "all", dim: "all", topN: 40,
    sort: { key: "absr", dir: "desc" }, search: "", category: "", minR: 0, sigOnly: true,
  };

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
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    setAttrs(node, attrs);
    append(node, children);
    return node;
  }
  function s(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    setAttrs(node, attrs);
    append(node, children);
    return node;
  }
  const $ = (id) => document.getElementById(id);
  const clear = (node) => { node.replaceChildren(); return node; };

  // ---------- formatting ----------
  const fmtInt = (n) => NF.format(n);
  function fmtNum(x, digits = 2) {
    if (x == null) return "–";
    const a = Math.abs(x);
    if (a !== 0 && (a < 0.005 || a >= 1e5)) return x.toExponential(1);
    return x.toFixed(digits);
  }
  function fmtP(p) {
    if (p == null) return "–";
    if (p === 0) return "< 1e-300";
    if (p < 0.001) return p.toExponential(1);
    return p.toFixed(3);
  }
  const signed = (x, digits = 2) => (x == null ? "–" : (x < 0 ? "−" : x > 0 ? "+" : "") + Math.abs(x).toFixed(digits));
  const levelName = (l) => (l === "all" ? (D.levels.length ? "all levels" : "all pairs") : `${D.meta.level_label} ${l}`);
  const levelHead = (l) => (l === "all" ? "All" : `${D.meta.level_label} ${l}`);
  const ctToken = (strategy, l) => `<|readability_${strategy}_${l}|>`;

  // ---------- data access ----------
  const key = (split, strategy, level) => `${split}|${strategy}|${level}`;
  const stratum = (split = state.split, strategy = state.strategy, level = state.level) => D.strata[key(split, strategy, level)];
  const levelsOf = (split, strategy) => D.levels.filter((l) => D.strata[key(split, strategy, l)]);

  // ---------- colour ----------
  let palette = {};
  function readPalette() {
    const cs = getComputedStyle(document.documentElement);
    const names = ["surface", "text-primary", "text-secondary", "text-muted", "series-1", "series-2", "series-3", "series-4", "series-5", "series-6", "series-7", "series-8", "div-neg", "div-mid", "div-pos"];
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
    return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * q,
      1.9779984951 * l - 2.428592205 * m + 0.4505937099 * q,
      0.0259040371 * l + 0.7827717662 * m - 0.808675766 * q];
  }
  function fromOklab([L, a, b]) {
    const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
    const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
    const q = (L - 0.0894841775 * a - 1.291485548 * b) ** 3;
    const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * q,
      -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * q,
      -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * q];
    const [r, g, bb] = rgb.map((c) => Math.round(Math.min(1, Math.max(0, toSrgb(c))) * 255));
    return `rgb(${r}, ${g}, ${bb})`;
  }
  function mix(hexA, hexB, t) {
    const A = toOklab(hexA), B = toOklab(hexB);
    return fromOklab(A.map((v, i) => v + (B[i] - v) * t));
  }
  // Diverging: blue = higher after simplification (r < 0), red = higher in the original (r > 0), grey midpoint.
  const divergingColor = (r) => mix(palette["div-mid"], palette[r >= 0 ? "div-pos" : "div-neg"], Math.min(1, Math.abs(r)));

  // ---------- tooltip ----------
  const tip = $("tooltip");
  function showTip(x, y, build) {
    clear(tip);
    build(tip);
    tip.hidden = false;
    const pad = 14, r = tip.getBoundingClientRect();
    let left = x + pad, top = y + pad;
    if (left + r.width > innerWidth - 8) left = x - r.width - pad;
    if (top + r.height > innerHeight - 8) top = y - r.height - pad;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  const tipRow = (color, value, name) => h("div", { class: "tt-row" },
    color ? h("span", { class: "key-line", style: `border-color:${color}` }) : null,
    h("strong", { text: value }), h("span", { class: "name", text: name }));

  // ---------- charts ----------
  function niceStep(max, count) {
    const raw = max / count, p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
    return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
  }
  const pct = (v, step) => `${(v * 100).toFixed(step * 100 < 1 ? 1 : 0)}%`;

  /** Overlaid KDE curves with a crosshair tooltip. series: [{name, color, values}] */
  function kdeChart(container, { series, height = 260, band = null, compact = false, showXAxis = true, label, xTitle }) {
    clear(container);
    const grid = D.kde_grid;
    const width = Math.max(compact ? 140 : 240, Math.floor(container.clientWidth || 600));
    if (width < 480 && xTitle) xTitle = xTitle.split(" (")[0];
    const m = compact
      ? { top: 4, right: 8, bottom: showXAxis ? 22 : 4, left: 8 }
      : { top: 12, right: 16, bottom: 44, left: 52 };
    const shown = series.filter((se) => se.values);
    const peak = Math.max(1e-9, ...shown.flatMap((se) => se.values));
    const yMax = peak * 1.08;
    const x0 = grid[0], x1 = grid[grid.length - 1];
    const x = (v) => m.left + ((v - x0) / (x1 - x0)) * (width - m.left - m.right);
    const y = (v) => height - m.bottom - (v / yMax) * (height - m.top - m.bottom);
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": label, tabindex: 0 });

    if (band) {
      svg.append(s("rect", { class: "band", x: x(band.from), y: m.top, width: x(band.to) - x(band.from), height: height - m.top - m.bottom }));
      if (!compact) svg.append(s("text", { class: "band-label", x: x(band.from) + 4, y: m.top + 12, text: band.label }));
    }
    if (!compact) {
      const step = niceStep(peak, 4);
      for (let v = step; v <= yMax; v += step) {
        svg.append(s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(v), y2: y(v) }));
        svg.append(s("text", { class: "axis-text", x: m.left - 8, y: y(v) + 4, "text-anchor": "end", text: pct(v, step) }));
      }
      svg.append(s("text", { class: "axis-title", x: 12, y: m.top + (height - m.top - m.bottom) / 2, "text-anchor": "middle",
        transform: `rotate(-90 12 ${m.top + (height - m.top - m.bottom) / 2})`, text: "Share of sentences per point" }));
    }
    svg.append(s("line", { class: "baseline", x1: m.left, x2: width - m.right, y1: y(0), y2: y(0) }));
    if (showXAxis) {
      for (let v = 0; v <= 100; v += 20) {
        svg.append(s("text", { class: "axis-text", x: x(v), y: height - m.bottom + 15, "text-anchor": "middle", text: v }));
      }
      if (!compact) svg.append(s("text", { class: "axis-title", x: m.left + (width - m.left - m.right) / 2, y: height - 6, "text-anchor": "middle", text: xTitle }));
    }
    for (const se of shown) {
      const pts = se.values.map((v, i) => `${x(grid[i]).toFixed(1)},${y(v).toFixed(1)}`);
      svg.append(s("path", { d: `M${x(x0)},${y(0)}L${pts.join("L")}L${x(x1)},${y(0)}Z`, fill: se.color, "fill-opacity": 0.1, stroke: "none" }));
      svg.append(s("path", { d: `M${pts.join("L")}`, fill: "none", stroke: se.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    }

    // hover layer: the crosshair snaps to the nearest grid point; one tooltip lists every series
    const hover = s("g", { style: "display:none" });
    const cross = s("line", { class: "crosshair", y1: m.top, y2: height - m.bottom });
    hover.append(cross);
    const dots = shown.map((se) => { const c = s("circle", { r: 4, fill: se.color, stroke: palette.surface, "stroke-width": 2 }); hover.append(c); return c; });
    svg.append(hover);
    const overlay = s("rect", { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: "transparent" });
    svg.append(overlay);
    let idx = grid.indexOf(50);
    function show(i, clientX, clientY) {
      idx = Math.max(0, Math.min(grid.length - 1, i));
      hover.style.display = "";
      cross.setAttribute("x1", x(grid[idx])); cross.setAttribute("x2", x(grid[idx]));
      shown.forEach((se, k) => { dots[k].setAttribute("cx", x(grid[idx])); dots[k].setAttribute("cy", y(se.values[idx])); });
      showTip(clientX, clientY, (t) => {
        t.append(h("div", { class: "tt-title", text: `READ-IT ${grid[idx]}` }));
        shown.forEach((se) => t.append(tipRow(se.color, pct(se.values[idx], 0.001), se.name)));
      });
    }
    const hide = () => { hover.style.display = "none"; hideTip(); };
    overlay.addEventListener("pointermove", (e) => {
      const r = svg.getBoundingClientRect();
      const v = x0 + ((e.clientX - r.left - m.left) / (width - m.left - m.right)) * (x1 - x0);
      show(Math.round(v - x0), e.clientX, e.clientY);
    });
    overlay.addEventListener("pointerleave", hide);
    svg.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") { if (e.key === "Escape") hide(); return; }
      e.preventDefault();
      const r = svg.getBoundingClientRect();
      const next = idx + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1);
      show(next, r.left + x(grid[Math.max(0, Math.min(grid.length - 1, next))]), r.top + m.top);
    });
    svg.addEventListener("blur", hide);
    container.append(svg);
  }

  function legend(container, items) {
    clear(container);
    for (const it of items) {
      container.append(h("span", { class: "legend-item" }, h("span", { class: "key-line", style: `border-color:${it.color}` }), it.name));
    }
  }

  function summaryTable(table, rows) {
    clear(table);
    table.append(h("thead", {}, h("tr", {},
      h("th", { text: "Sentences" }), h("th", { class: "num", text: "n" }), h("th", { class: "num", text: "Mean" }),
      h("th", { class: "num", text: "Median" }), h("th", { class: "num", text: "SD" }))));
    const body = h("tbody");
    for (const r of rows) {
      if (!r.summary) continue;
      body.append(h("tr", {},
        h("td", {}, h("span", { class: "key", style: `border-color:${r.color}` }), r.name),
        h("td", { class: "num", text: fmtInt(r.summary.n) }), h("td", { class: "num", text: fmtNum(r.summary.mean, 1) }),
        h("td", { class: "num", text: fmtNum(r.summary.median, 1) }), h("td", { class: "num", text: fmtNum(r.summary.sd, 1) })));
    }
    table.append(body);
  }

  function manovaText(mv) {
    if (!mv) return "not computed: too few pairs for the number of features";
    return `Pillai’s trace V = ${mv.V.toFixed(3)} (chance ≈ ${mv.chance.toFixed(3)}), F(${mv.df1}, ${fmtInt(mv.df2)}) = ${fmtNum(mv.F, 1)}, p ${mv.p === 0 ? "< 1e-300" : "= " + fmtP(mv.p)}`;
  }

  // ---------- static sections ----------
  function renderMeta() {
    const files = Object.entries(D.meta.sources).map(([f, h12]) => `${f} (${h12})`).join(", ");
    $("dataset-desc").textContent = D.meta.description;
    $("meta").textContent = `Generated ${D.meta.generated} from ${files}.`;
  }

  function renderCounts() {
    $("counts-title").textContent = D.levels.length ? `Pairs per ${D.meta.level_label === "CT" ? "control token" : "target level"}` : "Pairs";
    $("counts-sub").textContent = `${D.meta.counts_note} Within each slice the analyses keep one simplification per original sentence ` +
      "(the smaller “used” figure). Click a cell to inspect that slice below.";
    const table = clear($("counts"));
    const cols = ["all", ...D.levels];
    table.append(h("thead", {}, h("tr", {}, h("th", { scope: "col", text: "" }),
      cols.map((l) => h("th", { scope: "col", text: levelHead(l) })))));
    const body = h("tbody");
    for (const split of Object.keys(SPLITS)) {
      for (const strategy of Object.keys(STRATEGIES)) {
        if (!D.strata[key(split, strategy, "all")]) continue;
        const tr = h("tr", {}, h("th", { scope: "row", text: `${SPLITS[split]} · ${STRATEGIES[strategy]}` }));
        for (const level of cols) {
          const st = D.strata[key(split, strategy, level)];
          if (!st) { tr.append(h("td", { class: "none", text: "–" })); continue; }
          const selected = split === state.split && strategy === state.strategy && String(level) === String(state.level);
          const label = `${SPLITS[split]}, ${STRATEGIES[strategy]}, ${levelName(level)}: ${fmtInt(st.n_rows)} pairs` +
            (st.n_pairs !== st.n_rows ? `, ${fmtInt(st.n_pairs)} used` : "");
          tr.append(h("td", {}, h("button", {
            type: "button", class: [selected && "selected", st.n_pairs < MIN_PAIRS && "sparse"].filter(Boolean).join(" ") || null,
            "aria-pressed": selected ? "true" : "false", "aria-label": label, title: label,
            onclick: () => { setState({ split, strategy, level: level === "all" ? "all" : level }); $("readit-title").scrollIntoView({ behavior: "smooth" }); },
          }, fmtInt(st.n_rows), st.n_pairs !== st.n_rows ? h("span", { class: "used", text: `${fmtInt(st.n_pairs)} used` }) : null)));
        }
        body.append(tr);
      }
    }
    table.append(body);
  }

  const FIG1 = [
    { k: "original", name: "Original", color: "series-1" },
    { k: "most", name: "Most simplified", color: "series-2" },
    { k: "random", name: "Randomly selected", color: "series-3" },
    { k: "least", name: "Least simplified", color: "series-4" },
  ];
  function renderFig1() {
    const f = D.figure1;
    $("fig1-title").textContent = `Most, random and least simplified version of each original · ${D.meta.fig1_note}`;
    const counts = Object.keys(f.per_original).map(Number);
    $("fig1-sub").textContent = `${fmtInt(f.n_originals)} original sentences with ${Math.min(...counts)}–${Math.max(...counts)} simplifications each ` +
      `(${fmtInt(f.n_pairs)} pairs). As in the 2025 paper, the versions of each original are ranked by READ-IT global; ` +
      `the random version is drawn from the non-extreme ones. Not affected by the filters below.`;
    const series = FIG1.map((d) => ({ name: d.name, color: palette[d.color], values: f.kde[d.k], summary: f.summary[d.k] }));
    legend($("fig1-legend"), series);
    kdeChart($("fig1-chart"), { series, label: "READ-IT global distributions of original, most, randomly selected and least simplified sentences", xTitle: "READ-IT global (0 = most readable, 100 = least readable)" });
    summaryTable($("fig1-summary"), series);
    const t = clear($("fig1-manova"));
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "Original vs" }), h("th", { class: "num", text: "Pillai’s trace" }),
      h("th", { class: "num", text: "Chance" }), h("th", { class: "num", text: "F (df)" }), h("th", { class: "num", text: "p" }))));
    const body = h("tbody");
    for (const k of ["least", "random", "most"]) {
      const mv = f.manova[k], d = FIG1.find((x) => x.k === k);
      body.append(h("tr", {}, h("td", { text: d.name }),
        h("td", { class: "num", text: mv ? mv.V.toFixed(3) : "–" }), h("td", { class: "num", text: mv ? mv.chance.toFixed(3) : "–" }),
        h("td", { class: "num", text: mv ? `${fmtNum(mv.F, 1)} (${mv.df1}, ${fmtInt(mv.df2)})` : "–" }), h("td", { class: "num", text: mv ? fmtP(mv.p) : "–" })));
    }
    t.append(body);
  }

  function renderMethods() {
    const m = D.meta;
    const box = clear($("methods"));
    const code = (t) => h("code", { text: t });
    box.append(h("ul", {},
      h("li", {}, "Data: ", Object.keys(m.sources).map((f, i) => [i ? ", " : "", code(f)]), `: ${m.data_note} These statistics by `,
        code("scripts/build_webview_data.py"), ` on the master branch (seed ${m.seed}). File hashes (SHA-256, first 12 characters) are in the header.`),
      h("li", {}, `Slices: ${m.counts_note} `,
        "An original sentence can have several simplifications, so each slice keeps one per original, picked at random, as the 2025 analysis did; the test sets already have one per original. ",
        `Slices with fewer than ${MIN_PAIRS} pairs show counts and examples only.`),
      h("li", {}, "READ-IT scores are shown on a 0–100 scale (0 = most readable). Curves are Gaussian kernel density estimates (Scott’s bandwidth) evaluated from −10 to 110, so they spill past 0 and 100 as in the 2025 figure."),
      h("li", {}, "Per feature: Wilcoxon signed-rank test (zero differences discarded) and matched-pairs rank-biserial ", h("i", { text: "r" }),
        " = (W⁺ − W⁻) / (W⁺ + W⁻) on original minus simplified. ", h("i", { text: "r" }), " > 0 means the feature is higher in the original, the convention of the 2025 paper ",
        "(the CLiC-it 2026 paper uses the opposite sign). With thousands of pairs almost every p-value is tiny, so read the effect size. ",
        "Under each ", h("i", { text: "r" }), " is the number of pairs whose value changes at all: for sparse features ", h("i", { text: "r" }),
        " reaches ±1 when the few pairs that differ all go one way."),
      h("li", {}, "Spearman ", h("i", { text: "ρ" }), " correlates each feature’s original-minus-simplified difference with the difference in the selected READ-IT score (Figure 2 of the 2025 paper used READ-IT global)."),
      h("li", {}, "MANOVA: original and simplified sentences as two groups, as in the 2025 analysis, without ", m.manova_excluded.map((f, i) => [i ? " and " : "", code(f)]),
        "; collinear features are removed with a rank-revealing QR. With two groups Pillai’s trace has an exact F test. Its expected value under no difference, p / (N − 1), is shown as “chance”: it grows when a slice is small relative to the number of features, so compare slices against it."),
      h("li", {}, "Features come from Profiling-UD and are averages over the sentences of each text. ", code("n_sentences"),
        " is removed from the data because that averaging makes it always 1. Also left out because they never vary in this data: ",
        m.dropped_constant_features.length ? m.dropped_constant_features.map((f, i) => [i ? ", " : "", code(f)]) : "none", "."),
    ));
  }

  function renderCompare() {
    const table = clear($("compare"));
    const dims = ["all", "base", "lexical", "syntax"];
    table.append(h("thead", {},
      h("tr", {}, h("th", { rowspan: 2, text: "Dataset" }), h("th", { rowspan: 2, class: "num", text: "Pairs" }),
        h("th", { colspan: 4, class: "group", text: "READ-IT drop" }), h("th", { colspan: 3, class: "group", text: "Pillai’s trace" }),
        h("th", { colspan: COMPARE_FEATURES.length, class: "group" }, "Rank-biserial ", h("i", { text: "r" }))),
      h("tr", {}, dims.map((d) => h("th", { class: "num", text: DIMS[d] })),
        ["random", "most", "least"].map((k) => h("th", { class: "num", text: k })),
        COMPARE_FEATURES.map((f) => h("th", { class: "num", text: f })))));
    const body = h("tbody");
    for (const id of ORDER) {
      const ds = ALL[id], ov = ds.strata[ds.meta.overall_key];
      const drop = (d) => (ov.summary ? ov.summary[d].original.mean - ov.summary[d].simplified.mean : null);
      const mv = (k) => (ds.figure1.manova[k] ? ds.figure1.manova[k].V.toFixed(3) : "–");
      const rOf = (name) => {
        const j = ds.features.findIndex((f) => f.name === name);
        if (j < 0 || !ov.stats) return h("td", { class: "num muted", text: "–" });
        const sig = ov.stats.p[j] != null && ov.stats.p[j] < 0.05;
        return h("td", { class: `num${sig ? "" : " muted"}`, text: signed(ov.stats.r[j]) });
      };
      const tr = h("tr", { class: id === state.data ? "selected" : null, tabindex: 0, "aria-label": `Show ${ds.meta.title}`,
        onclick: () => setState({ data: id }), onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setState({ data: id }); } } },
        h("td", { text: ds.meta.title }), h("td", { class: "num", text: fmtInt(ov.n_pairs) }),
        dims.map((d) => h("td", { class: "num", text: fmtNum(drop(d), 1) })),
        ["random", "most", "least"].map((k) => h("td", { class: "num", text: mv(k) })),
        COMPARE_FEATURES.map(rOf));
      body.append(tr);
    }
    table.append(body);
  }

  // ---------- filtered sections ----------
  function fillSegmented(id, options, prop) {
    const box = clear($(id));
    for (const [value, label] of Object.entries(options)) {
      box.append(h("button", { type: "button", role: "radio", "data-value": value, "aria-checked": "false", onclick: () => setState({ [prop]: value }) }, label));
    }
  }
  function renderDatasetControls() {
    fillSegmented("split", SPLITS, "split");
    fillSegmented("strategy", STRATEGIES, "strategy");
    $("strategy-label").textContent = D.meta.strategy_label;
    $("level-label").textContent = !D.levels.length ? "Level" : D.meta.level_label === "CT" ? "Control token" : "Target level";
    for (const b of $("dataset").children) b.setAttribute("aria-checked", String(b.dataset.value === state.data));
  }

  function renderFilters() {
    for (const prop of ["split", "strategy"]) {
      for (const b of $(prop).children) b.setAttribute("aria-checked", String(b.dataset.value === state[prop]));
    }
    const sel = clear($("level"));
    const all = stratum(state.split, state.strategy, "all");
    sel.append(h("option", { value: "all", text: `All (${fmtInt(all.n_rows)} pairs)` }));
    for (const l of levelsOf(state.split, state.strategy)) {
      sel.append(h("option", { value: String(l), text: `${levelHead(l)} (${fmtInt(stratum(state.split, state.strategy, l).n_rows)})` }));
    }
    sel.value = String(state.level);
    $("dim").value = state.dim;
    const st = stratum();
    $("slice-summary").textContent = `${SPLITS[state.split]} · ${STRATEGIES[state.strategy]} · ${state.level === "all" ? levelName("all") : ctToken(state.strategy, state.level)} · ` +
      `${fmtInt(st.n_pairs)} pairs` + (st.n_pairs !== st.n_rows ? ` used of ${fmtInt(st.n_rows)}` : "");
  }

  const pairSeries = (kde, summary) => [
    { name: "Original", color: palette["series-1"], values: kde && kde.original, summary: summary && summary.original },
    { name: "Simplified", color: palette["series-2"], values: kde && kde.simplified, summary: summary && summary.simplified },
  ];
  const bandFor = (level) => (level !== "all" && state.dim === state.strategy
    ? { from: level, to: level + 10, label: `${ctToken(state.strategy, level)} range` } : null);

  function renderKdeMain() {
    const st = stratum();
    $("kde-title").textContent = `READ-IT ${DIMS[state.dim].toLowerCase()} · ${SPLITS[state.split]}, ${STRATEGIES[state.strategy]}, ${levelName(state.level)}`;
    let sub = `Original sentences and their simplifications in this slice (${fmtInt(st.n_pairs)} pairs).`;
    if (state.level !== "all") {
      sub += state.dim === state.strategy
        ? " The shaded band is the score range this control token was assigned from."
        : ` The control token was assigned from the READ-IT ${state.strategy} score; choose that score above to see its range.`;
    }
    $("kde-sub").textContent = sub;
    const series = pairSeries(st.kde && st.kde[state.dim], st.summary && st.summary[state.dim]);
    if (!st.kde) {
      clear($("kde-legend")); clear($("kde-summary"));
      clear($("kde-chart")).append(h("p", { class: "levels-empty", text: `Only ${st.n_pairs} pairs in this slice: too few for a distribution.` }));
      return;
    }
    legend($("kde-legend"), series);
    kdeChart($("kde-chart"), { series, band: bandFor(state.level), label: `READ-IT ${DIMS[state.dim]} distributions, original vs simplified`, xTitle: `READ-IT ${DIMS[state.dim].toLowerCase()} (0 = most readable, 100 = least readable)` });
    summaryTable($("kde-summary"), series);
  }

  function renderLevels() {
    $("levels-title").textContent = `READ-IT ${DIMS[state.dim].toLowerCase()} by ${D.meta.level_label === "CT" ? "control token" : "target level"} · ${SPLITS[state.split]}, ${STRATEGIES[state.strategy]}`;
    legend($("levels-legend"), pairSeries());
    const box = clear($("levels-chart"));
    const levels = levelsOf(state.split, state.strategy);
    if (!levels.length) box.append(h("p", { class: "levels-empty", text: "This slice has no levels: choose a strategy or dimension that has control tokens." }));
    levels.forEach((l, i) => {
      const st = stratum(state.split, state.strategy, l);
      const last = i === levels.length - 1;
      const row = h("div", {
        class: `levels-row${String(state.level) === String(l) ? " selected" : ""}`, role: "button", tabindex: 0,
        "aria-label": `Select ${levelHead(l)}`, onclick: () => setState({ level: l }),
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setState({ level: l }); } },
      }, h("div", { class: "levels-label" }, h("strong", { text: levelHead(l) }), `${fmtInt(st.n_pairs)} pairs`));
      const cell = h("div");
      row.append(cell);
      box.append(row);
      if (st.kde) {
        kdeChart(cell, { series: pairSeries(st.kde[state.dim]), height: last ? 78 : 56, compact: true, showXAxis: last, band: bandFor(l), label: `${levelHead(l)}: original vs simplified` });
      } else {
        cell.append(h("div", { class: "levels-empty", text: "too few pairs for a distribution" }));
      }
    });
  }

  function renderHeatLegend() {
    const stops = [-1, -0.66, -0.33, 0, 0.33, 0.66, 1].map((r, i) => `${divergingColor(r)} ${(i / 6) * 100}%`);
    const box = clear($("heat-legend"));
    box.append(h("span", { text: "−1 · higher after simplification" }),
      h("span", { class: "bar", style: `background: linear-gradient(90deg, ${stops.join(", ")})`, "aria-hidden": "true" }),
      h("span", { text: "+1 · higher in the original" }));
  }

  function renderHeatmap() {
    $("heat-title").textContent = `Effect sizes by ${D.levels.length ? (D.meta.level_label === "CT" ? "control token" : "target level") : "slice"} · ${SPLITS[state.split]}, ${STRATEGIES[state.strategy]}`;
    renderHeatLegend();
    const box = clear($("heat-chart"));
    const cols = ["all", ...levelsOf(state.split, state.strategy)].filter((l) => stratum(state.split, state.strategy, l).stats);
    const stats = cols.map((l) => stratum(state.split, state.strategy, l).stats);
    const sigR = (st, j) => (st.r[j] != null && st.p[j] != null && st.p[j] < 0.05 ? st.r[j] : null);
    const order = FEATURES.map((_, j) => ({ j, score: stats.reduce((a, st) => a + Math.abs(sigR(st, j) || 0), 0) / stats.length }))
      .sort((a, b) => b.score - a.score);
    const rows = state.topN ? order.slice(0, state.topN) : order;

    const labelW = 200, rowH = 18, cellH = 16, pillaiH = 74, headH = 36;
    const avail = Math.max(320, box.clientWidth || 800);
    const colW = Math.max(46, Math.min(76, Math.floor((avail - labelW - 8) / cols.length)));  // narrower screens scroll
    const width = labelW + colW * cols.length + 8;
    const top = pillaiH + headH;
    const height = top + rows.length * rowH + 4;
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": `Heatmap of rank-biserial effect sizes for ${rows.length} features across ${cols.length} control-token slices. The feature table below lists the same values.` });
    const cx = (c) => labelW + c * colW;

    const sel = cols.findIndex((l) => String(l) === String(state.level));
    if (sel >= 0) svg.append(s("rect", { class: "band", x: cx(sel), y: pillaiH - 4, width: colW, height: height - pillaiH + 4, rx: 4 }));

    // Pillai's trace per column, with a tick at the chance level
    const barTop = 16, barBase = pillaiH - 10;
    svg.append(s("text", { class: "axis-title", x: labelW - 8, y: barBase - 18, "text-anchor": "end", text: "Pillai’s trace" }));
    svg.append(s("text", { class: "axis-text", x: labelW - 8, y: barBase - 4, "text-anchor": "end", text: "(MANOVA, 0–1)" }));
    svg.append(s("line", { class: "baseline", x1: labelW, x2: width - 8, y1: barBase, y2: barBase }));
    const ph = (v) => barBase - v * (barBase - barTop);
    stats.forEach((st, c) => {
      const mv = st.manova, bw = Math.min(24, colW - 10), bx = cx(c) + (colW - bw) / 2;
      if (!mv) { svg.append(s("text", { class: "axis-text", x: cx(c) + colW / 2, y: barBase - 4, "text-anchor": "middle", text: "–" })); return; }
      const yTop = ph(mv.V), r = Math.min(4, (barBase - yTop) / 2);
      svg.append(s("path", { d: `M${bx},${barBase}V${yTop + r}Q${bx},${yTop} ${bx + r},${yTop}H${bx + bw - r}Q${bx + bw},${yTop} ${bx + bw},${yTop + r}V${barBase}Z`, fill: palette["series-1"] }));
      svg.append(s("line", { x1: bx - 3, x2: bx + bw + 3, y1: ph(mv.chance), y2: ph(mv.chance), stroke: palette["text-primary"], "stroke-width": 1.5 }));
      svg.append(s("text", { class: "axis-text", x: cx(c) + colW / 2, y: yTop - 4, "text-anchor": "middle", text: mv.V.toFixed(2) }));
      const hit = s("rect", { x: cx(c), y: barTop - 14, width: colW, height: barBase - barTop + 14, fill: "transparent" });
      hit.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, (t) => {
        t.append(h("div", { class: "tt-title", text: `MANOVA · ${levelName(cols[c])}` }), tipRow(null, mv.V.toFixed(3), "Pillai’s trace"),
          tipRow(null, mv.chance.toFixed(3), "expected by chance"), h("div", { class: "tt-note", text: `F(${mv.df1}, ${fmtInt(mv.df2)}) = ${fmtNum(mv.F, 1)}, p ${mv.p === 0 ? "< 1e-300" : "= " + fmtP(mv.p)}` }));
      }));
      hit.addEventListener("pointerleave", hideTip);
      svg.append(hit);
    });

    // column headers
    cols.forEach((l, c) => {
      const st = stratum(state.split, state.strategy, l);
      const isSel = c === sel;
      svg.append(s("text", { x: cx(c) + colW / 2, y: pillaiH + 14, "text-anchor": "middle", "font-size": 12, "font-weight": isSel ? 650 : 500,
        fill: isSel ? palette["text-primary"] : palette["text-secondary"], text: levelHead(l) }));
      svg.append(s("text", { class: "axis-text", x: cx(c) + colW / 2, y: pillaiH + 28, "text-anchor": "middle", "font-size": 10.5, text: fmtInt(st.n_pairs) }));
    });

    // cells
    const cells = s("g");
    rows.forEach(({ j }, i) => {
      const yy = top + i * rowH;
      svg.append(s("text", { class: "heat-label", x: labelW - 8, y: yy + cellH - 4, "text-anchor": "end", text: FEATURES[j] }));
      stats.forEach((st, c) => {
        const r = sigR(st, j);
        cells.append(s("rect", { class: "cell", x: cx(c) + 1, y: yy, width: colW - 2, height: cellH, rx: 2,
          fill: r == null ? "transparent" : divergingColor(r), "data-j": j, "data-c": c }));
      });
    });
    svg.append(cells);
    cells.addEventListener("pointermove", (e) => {
      const t = e.target;
      if (!t.dataset || t.dataset.j == null) return;
      const j = +t.dataset.j, c = +t.dataset.c, st = stats[c];
      showTip(e.clientX, e.clientY, (box2) => {
        box2.append(h("div", { class: "tt-title", text: `${FEATURES[j]} · ${levelName(cols[c])}` }),
          tipRow(null, signed(st.r[j]), "rank-biserial r"), tipRow(null, fmtP(st.p[j]), "Wilcoxon p"),
          h("div", { class: "tt-note", text: `${fmtInt(st.n_diff[j])} of ${fmtInt(stratum(state.split, state.strategy, cols[c]).n_pairs)} pairs differ` }),
          h("div", { class: "tt-note", text: `mean ${fmtNum(st.mean_o[j])} → ${fmtNum(st.mean_s[j])}` }),
          st.p[j] == null || st.p[j] >= 0.05 ? h("div", { class: "tt-note", text: "not significant" }) : null);
      });
    });
    cells.addEventListener("pointerleave", hideTip);
    cells.addEventListener("click", (e) => {
      if (e.target.dataset && e.target.dataset.c != null) setState({ level: cols[+e.target.dataset.c] });
    });
    box.append(svg);
  }

  const TABLE_COLS = [
    { key: "name", label: "Feature" },
    { key: "mo", label: "Original mean (SD)", num: true },
    { key: "ms", label: "Simplified mean (SD)", num: true },
    { key: "absr", label: "r", num: true, title: "rank-biserial r, sorted by |r|" },
    { key: "p", label: "p (Wilcoxon)", num: true },
    { key: "absrho", label: "ρ", num: true, title: "Spearman ρ with the READ-IT difference, sorted by |ρ|" },
    { key: "rhop", label: "p (ρ)", num: true },
  ];

  function tableRows() {
    const st = stratum().stats;
    if (!st) return [];
    const q = state.search.trim().toLowerCase();
    return FEATURES.map((name, j) => ({
      j, name, cat: D.features[j].category, mo: st.mean_o[j], so: st.sd_o[j], ms: st.mean_s[j], ss: st.sd_s[j],
      r: st.r[j], p: st.p[j], nd: st.n_diff[j], rho: st.rho[state.dim][j], rhop: st.rho_p[state.dim][j],
    })).filter((row) => (!q || row.name.toLowerCase().includes(q))
      && (!state.category || row.cat === state.category)
      && (!state.sigOnly || (row.p != null && row.p < 0.05))
      && (row.r == null ? state.minR <= 0 : Math.abs(row.r) >= state.minR));
  }

  function sortRows(rows) {
    const { key: k, dir } = state.sort;
    const val = (row) => (k === "absr" ? (row.r == null ? -1 : Math.abs(row.r)) : k === "absrho" ? (row.rho == null ? -1 : Math.abs(row.rho))
      : k === "p" ? (row.p == null ? 2 : row.p) : k === "rhop" ? (row.rhop == null ? 2 : row.rhop) : row[k]);
    const sign = dir === "asc" ? 1 : -1;
    return rows.sort((a, b) => {
      const va = val(a), vb = val(b);
      if (typeof va === "string") return sign * va.localeCompare(vb);
      return sign * ((va ?? -Infinity) - (vb ?? -Infinity));
    });
  }

  function rBar(r) {
    const w = 56, hgt = 10, mid = w / 2;
    const svg = s("svg", { width: w, height: hgt, "aria-hidden": "true" });
    svg.append(s("line", { x1: mid, x2: mid, y1: 0, y2: hgt, stroke: palette["text-muted"], "stroke-width": 1 }));
    if (r != null && r !== 0) {
      const len = Math.abs(r) * (mid - 1);
      svg.append(s("rect", { x: r > 0 ? mid : mid - len, y: 1, width: Math.max(1, len), height: hgt - 2, rx: 2, fill: divergingColor(r) }));
    }
    return svg;
  }

  function renderTable() {
    const st = stratum();
    $("table-title").textContent = `Features · ${SPLITS[state.split]}, ${STRATEGIES[state.strategy]}, ${levelName(state.level)} · ${fmtInt(st.n_pairs)} pairs`;
    const table = clear($("features"));
    if (!st.stats) {
      $("manova-line").textContent = "";
      $("row-count").textContent = "";
      table.append(h("tbody", {}, h("tr", {}, h("td", { text: `Only ${st.n_pairs} pairs in this slice: too few for the tests.` }))));
      return;
    }
    $("manova-line").textContent = `MANOVA, original vs simplified: ${manovaText(st.stats.manova)}.`;
    const head = h("tr");
    for (const c of TABLE_COLS) {
      const sorted = state.sort.key === c.key;
      const label = c.key === "absrho" ? `ρ with Δ READ-IT ${DIMS[state.dim].toLowerCase()}` : c.label;
      head.append(h("th", { class: c.num ? "num" : null, scope: "col", "aria-sort": sorted ? (state.sort.dir === "asc" ? "ascending" : "descending") : "none", title: c.title || null },
        h("button", { type: "button", onclick: () => {
          state.sort = { key: c.key, dir: sorted && state.sort.dir === "desc" ? "asc" : sorted ? "desc" : (c.num ? "desc" : "asc") };
          renderTable();
        } }, c.key === "absr" ? h("i", { text: "r" }) : label)));
    }
    table.append(h("thead", {}, head));
    const rows = sortRows(tableRows());
    const body = h("tbody");
    for (const row of rows) {
      const sig = row.p != null && row.p < 0.05;
      body.append(h("tr", {},
        h("td", { class: "feature" }, row.name, h("span", { class: "cat", text: row.cat })),
        h("td", { class: "num", text: `${fmtNum(row.mo)} (±${fmtNum(row.so)})` }),
        h("td", { class: "num", text: `${fmtNum(row.ms)} (±${fmtNum(row.ss)})` }),
        h("td", { class: `num${sig ? "" : " muted"}` }, h("span", { class: "rbar" }, rBar(sig ? row.r : null), signed(row.r)),
          h("span", { class: "nd", text: `${fmtInt(row.nd)} of ${fmtInt(st.n_pairs)} differ` })),
        h("td", { class: `num${sig ? "" : " muted"}`, text: fmtP(row.p) }),
        h("td", { class: "num", text: signed(row.rho) }),
        h("td", { class: "num", text: fmtP(row.rhop) })));
    }
    table.append(body);
    $("row-count").textContent = `${rows.length} of ${FEATURES.length} features`;
  }

  function downloadCsv() {
    const rows = sortRows(tableRows());
    const esc = (v) => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const header = ["feature", "category", "mean_original", "sd_original", "mean_simplified", "sd_simplified", "rank_biserial_r", "p_wilcoxon", "pairs_that_differ", `spearman_rho_readit_${state.dim}`, "p_spearman"];
    const lines = [header.join(","), ...rows.map((r) => [r.name, r.cat, r.mo, r.so, r.ms, r.ss, r.r, r.p, r.nd, r.rho, r.rhop].map(esc).join(","))];
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    const a = h("a", { href: URL.createObjectURL(blob), download: `profile_${state.data}_${state.split}_${state.strategy}_${state.level}.csv` });
    document.body.append(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  function renderExamples() {
    const st = stratum();
    $("examples-sub").textContent = `${st.examples.length} pairs drawn at random from ${SPLITS[state.split].toLowerCase()}, ${STRATEGIES[state.strategy]}, ${levelName(state.level)}. ` +
      `Scores are READ-IT ${DIMS[state.dim].toLowerCase()}.`;
    const box = clear($("examples"));
    for (const ex of st.examples) {
      box.append(h("article", { class: "example" },
        h("div", { class: "tag", text: `Original · READ-IT ${fmtNum(ex.readit_o[state.dim], 1)}` }), h("p", { text: ex.original }),
        h("div", { class: "tag", text: `Simplified · READ-IT ${fmtNum(ex.readit_s[state.dim], 1)}` }), h("p", { text: ex.simplification })));
    }
  }

  // ---------- part 2: model generation analysis ----------
  const G = window.GENERATION_DATA;
  // One colour per model, fixed, so toggling a line in the legend never repaints the others. Six lines cannot all be
  // told apart, so the curve starts with the CLiC-it model and our two latest models (blue, magenta, green stay apart
  // pairwise in both themes); the legend shows the others. The tables always list every model.
  const GEN_COLORS = { "clicit2026_0.6B": "series-1", targets: "series-2", source_targets: "series-8", v2_targets: "series-4",
                       v3_targets: "series-5", v4_targets: "series-3", v5_targets: "series-7", v6_targets: "series-6" };
  const GEN_SPARE = ["series-1", "series-2", "series-3", "series-4", "series-5", "series-6", "series-7", "series-8"];
  const GEN_DEFAULT_SHOWN = 2;
  const GEN_DIMS = { base: "Base", lexical: "Lexical", syntax: "Syntax" };
  const gen = { dim: "lexical", shown: null };
  const pctOf = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
  const genColor = (id, i) => palette[GEN_COLORS[id] || GEN_SPARE[i % GEN_SPARE.length]];
  const genModels = () => G.meta.models.map((m, i) => ({ ...m, color: genColor(m.id, i) })).filter((m) => G.main[gen.dim] && G.main[gen.dim][m.id]);

  function genShown() {
    if (!gen.shown) {
      const clic = G.meta.models.filter((m) => m.id.startsWith("clicit")).map((m) => m.id);
      const ours = G.meta.models.filter((m) => !m.id.startsWith("clicit")).map((m) => m.id);
      gen.shown = new Set([...clic, ...ours.slice(-GEN_DEFAULT_SHOWN)]);
    }
    return gen.shown;
  }

  function genLegend(container) {
    clear(container);
    const shown = genShown();
    for (const mod of genModels()) {
      const on = shown.has(mod.id);
      container.append(h("button", { type: "button", class: "legend-item legend-toggle", "aria-pressed": String(on),
        title: on ? "Hide this model’s line" : "Show this model’s line",
        onclick: () => { on ? shown.delete(mod.id) : shown.add(mod.id); genLegend(container); genCurve($("gen-curve-chart")); } },
        h("span", { class: "key-line", style: `border-color:${mod.color}` }), mod.label));
    }
  }

  function genCurve(container) {
    clear(container);
    const shown = genShown();
    const models = genModels().filter((m) => shown.has(m.id));
    const levels = [...new Set(models.flatMap((m) => Object.keys(G.main[gen.dim][m.id].levels).map(Number)))].sort((a, b) => a - b);
    if (!levels.length) return;
    const width = Math.max(280, Math.floor(container.clientWidth || 720)), height = 340;
    const m = { top: 12, right: 16, bottom: 46, left: 52 };
    const xMax = levels[levels.length - 1] + 10;
    const x = (v) => m.left + (v / xMax) * (width - m.left - m.right);
    const y = (v) => height - m.bottom - (Math.max(0, Math.min(100, v)) / 100) * (height - m.top - m.bottom);
    const svg = s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, style: "max-width:100%;height:auto", role: "img", tabindex: 0,
      "aria-label": `Achieved READ-IT ${gen.dim} against the requested level, one line per model. The tables below list the same values.` });
    for (let v = 20; v <= 100; v += 20) {
      svg.append(s("line", { class: "gridline", x1: m.left, x2: width - m.right, y1: y(v), y2: y(v) }));
      svg.append(s("text", { class: "axis-text", x: m.left - 8, y: y(v) + 4, "text-anchor": "end", text: v }));
    }
    svg.append(s("line", { class: "baseline", x1: m.left, x2: width - m.right, y1: y(0), y2: y(0) }));
    for (const L of levels) {   // where an output should land: [L, L + 10)
      svg.append(s("rect", { class: "band", x: x(L) + 1, width: x(L + 10) - x(L) - 2, y: y(L + 10), height: y(L) - y(L + 10), rx: 2 }));
      svg.append(s("text", { class: "axis-text", x: x(L + 5), y: height - m.bottom + 15, "text-anchor": "middle", text: L }));
    }
    svg.append(s("text", { class: "axis-title", x: m.left + (width - m.left - m.right) / 2, y: height - 8, "text-anchor": "middle", text: `Requested ${gen.dim} level` }));
    svg.append(s("text", { class: "axis-title", x: 12, y: m.top + (height - m.top - m.bottom) / 2, "text-anchor": "middle",
      transform: `rotate(-90 12 ${m.top + (height - m.top - m.bottom) / 2})`, text: `Achieved READ-IT ${gen.dim}` }));
    const step = Math.min(2.4, 6 / Math.max(1, models.length));
    models.forEach((mod, i) => {
      const lv = G.main[gen.dim][mod.id].levels, off = (i - (models.length - 1) / 2) * step;
      const pts = levels.filter((L) => lv[L] && lv[L].achieved_mean != null).map((L) => ({ L, cx: x(L + 5 + off), ...lv[L] }));
      for (const p of pts) {
        const [q1, , q3] = p.achieved_quartiles;
        svg.append(s("line", { x1: p.cx, x2: p.cx, y1: y(q1), y2: y(q3), stroke: mod.color, "stroke-width": 2, "stroke-linecap": "round", opacity: 0.55 }));
      }
      svg.append(s("path", { d: `M${pts.map((p) => `${p.cx.toFixed(1)},${y(p.achieved_mean).toFixed(1)}`).join("L")}`, fill: "none",
        stroke: mod.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      for (const p of pts) {
        const few = (p.scored || 0) < 30;
        svg.append(s("circle", { cx: p.cx, cy: y(p.achieved_mean), r: few ? 3.5 : 4, fill: few ? palette.surface : mod.color,
          stroke: few ? mod.color : palette.surface, "stroke-width": 2 }));
      }
    });
    for (const L of levels) {   // hover: one column per requested level
      const hit = s("rect", { x: x(L), y: m.top, width: x(L + 10) - x(L), height: height - m.top - m.bottom, fill: "transparent" });
      hit.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, (t) => {
        t.append(h("div", { class: "tt-title", text: `Requested ${gen.dim} ${L} (target ${L}–${L + 10})` }));
        for (const mod of models) {
          const v = G.main[gen.dim][mod.id].levels[L];
          if (!v) continue;
          t.append(tipRow(mod.color, fmtNum(v.achieved_mean, 1), `${mod.label} · mean`),
            h("div", { class: "tt-note", text: `median ${fmtNum(v.achieved_quartiles[1], 1)}, middle half ${fmtNum(v.achieved_quartiles[0], 1)}–${fmtNum(v.achieved_quartiles[2], 1)}; inside ${pctOf(v.in_level)}, RMSE ${fmtNum(v.rmse, 1)}, copies of the original ${pctOf(v.copy)}, n ${fmtInt(v.scored)}` }));
        }
      }));
      hit.addEventListener("pointerleave", hideTip);
      svg.append(hit);
    }
    container.append(svg);
  }

  function genSummary() {
    const models = genModels(), free = Object.keys(GEN_DIMS).filter((d) => d !== gen.dim);
    $("gen-summary-title").textContent = `${GEN_DIMS[gen.dim]} requests · overall`;
    $("gen-summary-sub").textContent = `Every request of the ${gen.dim} test set: each original at every level below its own ${gen.dim} level, the other two dimensions left free. ` +
      `Inside: share of outputs inside the requested level. Bias: achieved minus requested, mean. ρ: Spearman between requested and achieved, over all requests and within each original. ` +
      `Δ: mean change of READ-IT from the original (points, negative = simpler); the requested dimension should move, the free ones are what comes along. Meaning is measured against the original.`;
    const t = clear($("gen-summary"));
    t.append(h("thead", {}, h("tr", {}, ["Model", "Requests", "RMSE", "Inside", "Bias", "ρ", "ρ within",
      `Δ ${gen.dim}`, ...free.map((d) => `Δ ${d}`), "BERTScore", "SBERT", "Copies"].map((c, i) => h("th", { class: i ? "num" : null, text: c })))));
    const body = h("tbody");
    for (const mod of models) {
      const r = G.main[gen.dim][mod.id], o = r.overall;
      body.append(h("tr", {}, h("td", {}, h("span", { class: "key", style: `border-color:${mod.color}` }), mod.label),
        h("td", { class: "num", text: fmtInt(r.requests) }), h("td", { class: "num", text: fmtNum(o.rmse, 1) }),
        h("td", { class: "num", text: pctOf(o.in_level) }), h("td", { class: "num", text: signed(o.bias, 1) }),
        h("td", { class: "num", text: fmtNum(o.spearman) }), h("td", { class: "num", text: fmtNum(o.within_original_spearman) }),
        h("td", { class: "num", text: signed(o.moved_requested, 1) }), ...free.map((d) => h("td", { class: "num", text: signed(o.moved_free[d], 1) })),
        h("td", { class: "num", text: fmtNum(o.bertscore_f1, 3) }), h("td", { class: "num", text: fmtNum(o.sbert_cosine, 3) }),
        h("td", { class: "num", text: pctOf(o.copy) })));
    }
    t.append(body);
  }

  function genLevels() {
    const models = genModels();
    const levels = [...new Set(models.flatMap((m) => Object.keys(G.main[gen.dim][m.id].levels).map(Number)))].sort((a, b) => a - b);
    $("gen-levels-title").textContent = `${GEN_DIMS[gen.dim]} requests · by level`;
    const t = clear($("gen-levels"));
    t.append(h("thead", {},
      h("tr", {}, h("th", { rowspan: 2, text: "Requested" }), h("th", { rowspan: 2, class: "num", text: "Requests" }),
        models.map((m) => h("th", { colspan: 3, class: "group" }, h("span", { class: "key", style: `border-color:${m.color}` }), m.label))),
      h("tr", {}, models.flatMap(() => ["Mean", "RMSE", "Inside"].map((c) => h("th", { class: "num", text: c }))))));
    const body = h("tbody");
    for (const L of levels) {
      const n = Math.max(...models.map((m) => (G.main[gen.dim][m.id].levels[L] || {}).scored || 0));
      body.append(h("tr", {}, h("td", { text: `${L}–${L + 10}` }), h("td", { class: "num", text: fmtInt(n) }),
        models.flatMap((m) => {
          const v = G.main[gen.dim][m.id].levels[L];
          return v ? [h("td", { class: "num", text: fmtNum(v.achieved_mean, 1) }), h("td", { class: "num", text: fmtNum(v.rmse, 1) }),
            h("td", { class: "num" }, h("span", { class: "in", text: pctOf(v.in_level) }))]
            : [0, 1, 2].map(() => h("td", { class: "num", text: "–" }));
        })));
    }
    t.append(body);
  }

  function genJoint() {
    const t = clear($("gen-joint"));
    const models = G.meta.models.map((m, i) => ({ ...m, color: genColor(m.id, i) })).filter((m) => G.joint[m.id]);
    $("gen-joint-sub").textContent = "Requests with all three targets set, at most two per test original: “achievable” uses the levels of one of the original’s own IMPaCTS " +
      "simplifications that lowered all three dimensions; “typical” uses the levels of a training pair whose original starts at the same levels. " +
      "All three: share of outputs inside all three requested levels at once; within one level: all three within one level below or above the request; Base, Lexical, Syntax: share inside that level alone. " +
      "The CLiC-it 2026 model takes one token at a time and is not part of this test.";
    if (!models.length) { t.append(h("tbody", {}, h("tr", {}, h("td", { text: "No joint results yet." })))); return; }
    t.append(h("thead", {}, h("tr", {}, ["Model", "Type", "Requests", "All three", "Within one level", "Base", "Lexical", "Syntax", "BERTScore", "SBERT"]
      .map((c, i) => h("th", { class: i > 1 ? "num" : null, text: c })))));
    const body = h("tbody");
    for (const mod of models) {
      for (const type of ["all", "achievable", "typical"]) {
        const r = G.joint[mod.id][type];
        if (!r) continue;
        body.append(h("tr", {}, h("td", {}, type === "all" ? [h("span", { class: "key", style: `border-color:${mod.color}` }), mod.label] : ""),
          h("td", { text: type }), h("td", { class: "num", text: fmtInt(r.requests) }),
          h("td", { class: "num" }, h("span", { class: "in", text: `${(100 * r.all_three_in_level).toFixed(1)}%` })), h("td", { class: "num", text: pctOf(r.all_three_within_one_level) }),
          ...["base", "lexical", "syntax"].map((d) => h("td", { class: "num", text: pctOf(r.per_dimension[d].in_level) })),
          h("td", { class: "num", text: fmtNum(r.bertscore_f1, 3) }), h("td", { class: "num", text: fmtNum(r.sbert_cosine, 3) })));
      }
    }
    t.append(body);
  }

  function genExample() {
    const ex = G.examples[gen.dim], t = clear($("gen-example")), models = genModels();
    if (!ex) { $("gen-example-sub").textContent = ""; return; }
    $("gen-example-sub").textContent = `A ${gen.dim} test original (READ-IT ${gen.dim} ${fmtNum(ex.original_score, 1)}) and what each model wrote at every requested level. ` +
      `Under each output: its achieved READ-IT ${gen.dim}. Sampling at temperature 0.7, so another run would differ.`;
    t.append(h("thead", {}, h("tr", {}, h("th", { text: "Level" }), models.map((m) => h("th", {}, h("span", { class: "key", style: `border-color:${m.color}` }), m.label)))));
    const body = h("tbody");
    body.append(h("tr", {}, h("td", { class: "level", text: "Original" }), h("td", { colspan: models.length, text: ex.original })));
    for (const row of ex.rows) {
      body.append(h("tr", {}, h("td", { class: "level", text: `${row.level}–${row.level + 10}` }), models.map((m) => {
        const o = row[m.id];
        if (!o) return h("td", { text: "–" });
        const inside = o.score != null && o.score >= row.level && o.score < row.level + 10;
        return h("td", {}, o.text, h("span", { class: "score", text: `${gen.dim} ${o.score == null ? "–" : fmtNum(o.score, 1)}${inside ? " · inside" : ""}` }));
      })));
    }
    t.append(body);
  }

  function genMethods() {
    const box = clear($("gen-methods")), code = (x) => h("code", { text: x });
    box.append(h("ul", {},
      h("li", {}, "Models: Qwen3-0.6B-Base fine-tuned with LoRA on dataset v1 (", code("data/v1/"), " on master), one with three target tokens per request and one also given the original’s three READ-IT levels; the target-token model trained the same way on dataset v2 (", code("data/v2/"), ": v1 plus 11,752 pairs with larger lexical drops), on dataset v3 (", code("data/v3/"), ": v2 with same-level base and syntax targets) and on dataset v4 (", code("data/v4/"), ": v3 with a syntax target on every pair whose syntax did not rise) and on dataset v5 (", code("data/v5/"), ": v3 plus 989 identity pairs at the top of the syntax scale, at the CLiC-it 2026 data’s share) and on dataset v6 (", code("data/v6/"), ": the same with 240 identity pairs, the CLiC-it share among examples that give the syntax token alone); and the CLiC-it 2026 Qwen3-0.6B, whose existing outputs are re-scored."),
      h("li", {}, "Requests: each test original at every level below its own level in the test set’s dimension, the other two dimensions left free; sampling at temperature 0.7, top-p 1.0, up to 128 new tokens."),
      h("li", {}, "READ-IT is computed exactly as the IMPaCTS labels (t2k with its default configuration, one text at a time, the custom parser under stanza 1.10, mean over sentences): re-scoring the 3,000 test originals and 3,000 references reproduces every label. Outputs, originals and requested levels share one scale."),
      h("li", {}, "RMSE is measured against the bottom of the requested level, as in CLiC-it 2026, so an output inside the level still scores about 5; the share inside the level is the stricter, clearer measure."),
      h("li", {}, "t2k scores only sentences of more than 5 tokens, so an output made only of shorter sentences has no READ-IT score (0.1–1.5% of the outputs of each set); these are left out of the control measures."),
      h("li", {}, "Meaning preservation, against the original: BERTScore F1 (bert-base-multilingual-cased, layer 9) and SBERT cosine (distiluse-base-multilingual-cased-v1)."),
      h("li", {}, "Code: ", code("data/v1/experiments/"), " on master; this section’s numbers by ", code("scripts/build_generation_webview.py"), ".")));
  }

  function renderGeneration() {
    if (!G) return;
    $("generation-part").hidden = false;
    $("gen-lede").textContent = "Qwen3-0.6B models fine-tuned on datasets v1 to v6, asked to simplify the test originals to every READ-IT level below the original’s own, " +
      "compared with the CLiC-it 2026 Qwen3-0.6B. The curves follow the CLiC-it 2026 figures: requested level against achieved READ-IT.";
    $("gen-meta").textContent = `Generated ${G.meta.generated}.`;
    for (const b of $("gen-dim").children) b.setAttribute("aria-checked", String(b.dataset.value === gen.dim));
    const models = genModels();
    $("gen-curve-title").textContent = `${GEN_DIMS[gen.dim]} requests · achieved against requested`;
    $("gen-curve-sub").textContent = `Each point is the mean achieved READ-IT ${gen.dim} of the outputs at that requested level; the vertical line spans their middle half. ` +
      "The shaded staircase is where an output should land. Hollow points rest on fewer than 30 outputs (few originals sit that high). Hover a level for the numbers. " +
      "The CLiC-it model and our two latest models are drawn first; click a model in the legend to show or hide its line. The tables below list every model.";
    genLegend($("gen-curve-legend"));
    genCurve($("gen-curve-chart"));
    genSummary();
    genLevels();
    genJoint();
    genExample();
    genMethods();
  }

  function initGeneration() {
    if (!G) return;
    const box = $("gen-dim");
    for (const [value, label] of Object.entries(GEN_DIMS)) {
      box.append(h("button", { type: "button", role: "radio", "data-value": value, "aria-checked": "false",
        onclick: () => { gen.dim = value; renderGeneration(); writeHash(); } }, label));
    }
  }

  // ---------- state, URL, wiring ----------
  function normalise() {
    if (!ALL[state.data]) state.data = ORDER[0];
    if (D !== ALL[state.data]) useDataset(state.data);
    if (!SPLITS[state.split]) state.split = Object.keys(SPLITS)[0];
    if (!STRATEGIES[state.strategy] || !stratum(state.split, state.strategy, "all")) {
      state.strategy = Object.keys(STRATEGIES).find((k) => stratum(state.split, k, "all"));
    }
    if (!DIMS[state.dim]) state.dim = "all";
    if (state.level !== "all") {
      state.level = Number(state.level);
      if (!stratum()) state.level = "all";
    }
  }
  function writeHash() {
    const p = new URLSearchParams({ data: state.data, split: state.split, strategy: state.strategy, level: state.level, score: state.dim, gen: gen.dim });
    history.replaceState(null, "", `#${p}`);
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (p.get("data")) state.data = p.get("data");
    if (p.get("split")) state.split = p.get("split");
    if (p.get("strategy")) state.strategy = p.get("strategy");
    if (p.get("level")) state.level = p.get("level") === "all" ? "all" : Number(p.get("level"));
    if (p.get("score")) state.dim = p.get("score");
    if (GEN_DIMS[p.get("gen")]) gen.dim = p.get("gen");
  }
  function setState(patch) {
    Object.assign(state, patch);
    render();
  }
  let shownData = null;
  function render() {
    normalise();
    readPalette();
    if (shownData !== state.data) {
      shownData = state.data;
      renderDatasetControls();
      renderMeta();
      renderMethods();
      renderFig1();
    }
    renderCompare();
    renderFilters();
    renderCounts();
    renderKdeMain();
    renderLevels();
    renderHeatmap();
    renderTable();
    renderExamples();
    writeHash();
  }
  function renderAll() {
    shownData = null;
    render();
    renderGeneration();
  }

  // Arrow keys move through a segmented control; the options are read at key time, since they change with the dataset.
  function segmentedKeys(id, prop, optionsOf) {
    const box = $(id);
    box.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const values = Object.keys(optionsOf()), i = values.indexOf(state[prop]);
      const next = values[(i + (e.key === "ArrowRight" ? 1 : values.length - 1)) % values.length];
      setState({ [prop]: next });
      box.querySelector(`[data-value="${next}"]`).focus();
    });
  }

  const THEMES = ["auto", "light", "dark"];
  let theme = "auto";
  function applyTheme(next) {
    theme = THEMES.includes(next) ? next : "auto";
    if (theme === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    $("theme-toggle").textContent = `Theme: ${theme}`;
  }
  function initTheme() {
    let saved = "auto";
    try { saved = localStorage.getItem("webview-theme") || "auto"; } catch (e) { /* storage unavailable */ }
    applyTheme(saved);
    $("theme-toggle").addEventListener("click", () => {
      applyTheme(THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length]);
      try { localStorage.setItem("webview-theme", theme); } catch (e) { /* storage unavailable */ }
      renderAll();
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  }

  function init() {
    initTheme();
    if (!ORDER.length) { document.querySelector("main").textContent = "No dataset loaded."; return; }
    useDataset(state.data);
    fillSegmented("dataset", Object.fromEntries(ORDER.map((k) => [k, ALL[k].meta.label])), "data");
    segmentedKeys("dataset", "data", () => Object.fromEntries(ORDER.map((k) => [k, 1])));
    segmentedKeys("split", "split", () => SPLITS);
    segmentedKeys("strategy", "strategy", () => STRATEGIES);
    for (const [v, label] of Object.entries(DIMS)) $("dim").append(h("option", { value: v, text: label }));
    $("level").addEventListener("change", (e) => setState({ level: e.target.value === "all" ? "all" : Number(e.target.value) }));
    $("dim").addEventListener("change", (e) => setState({ dim: e.target.value }));
    $("topn").addEventListener("change", (e) => { state.topN = Number(e.target.value); renderHeatmap(); });
    const cat = $("category");
    cat.append(h("option", { value: "", text: "all" }), ...D.categories.map((c) => h("option", { value: c, text: c })));
    cat.addEventListener("change", (e) => { state.category = e.target.value; renderTable(); });
    $("search").addEventListener("input", (e) => { state.search = e.target.value; renderTable(); });
    $("min-r").addEventListener("input", (e) => { state.minR = Math.max(0, Number(e.target.value) || 0); renderTable(); });
    $("sig-only").addEventListener("change", (e) => { state.sigOnly = e.target.checked; renderTable(); });
    $("download").addEventListener("click", downloadCsv);
    readHash();
    initGeneration();
    renderAll();
    let timer;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  }

  init();
})();
