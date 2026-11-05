// 评测结果看板：指标卡、航司×票种热力图、失败原因条形图、失败用例下钻、历史对比
import { icon } from "../icons.js";
import { renderTrace } from "../trace.js";
import { esc, empty, drawer, ms, pct, toastError, outcomeChip, AIRLINES, PRODUCTS, INTENT } from "../ui.js";

// [key, 标签, 格式, 越高越好?]
const METRICS = [
  ["correctness", "方案正确率", "pct", true, "outcome、金额全对且没有 forbid 错误"],
  ["tool_acc", "工具调用准确率", "pct", true, "工具序列与期望一致且 Schema 全部通过"],
  ["policy_hit", "政策命中率", "pct", true, "期望规则全部出现在前 3 条"],
  ["human_rate", "转人工率", "pct", false, "结局为 HUMAN 的比例"],
  ["latency_avg_ms", "平均延迟", "ms", false, ""],
  ["latency_p95_ms", "P95 延迟", "ms", false, ""],
  ["tokens_avg", "平均 Token", "int", false, ""],
];
const fmt = (v, f) => (v === null || v === undefined ? "—" : f === "pct" ? pct(v) : f === "ms" ? ms(v) : Math.round(v).toLocaleString());

function metricCards(s) {
  return `<div class="metrics">${METRICS.map(([k, label, f, up, help]) => {
    const v = s[k];
    const good = f === "pct" && k === "correctness" ? (v >= 0.9 ? "ok" : "bad") : "";
    return `<div class="metric ${good}" title="${esc(help)}"><div class="m-label">${esc(label)}</div><div class="m-val num">${fmt(v, f)}</div>
      ${k === "correctness" ? `<div class="m-sub">${v >= 0.9 ? `<span class="status ok">${icon("check", "sm")}达到 90% 基线</span>` : `<span class="status bad">${icon("x", "sm")}低于 90% 基线</span>`}</div>` : help ? `<div class="m-sub faint">${esc(help)}</div>` : ""}</div>`;
  }).join("")}<div class="metric"><div class="m-label">用例数</div><div class="m-val num">${s.n ?? 0}</div><div class="m-sub faint">本次运行</div></div></div>`;
}

/** 航司 × 票种：由逐条结果 + 用例元数据计算 */
function heatmap(results, caseMap) {
  const cell = {};
  for (const r of results) {
    const c = caseMap[r.case_id] || {};
    const a = c.airline || "—", p = c.fare_product || "—";
    const x = (cell[`${a}|${p}`] ||= { n: 0, ok: 0 });
    x.n++; if (r.passed) x.ok++;
  }
  const airlines = [...Object.keys(AIRLINES), "—"].filter((a) => Object.keys(cell).some((k) => k.startsWith(`${a}|`)));
  const prods = [...Object.keys(PRODUCTS), "—"].filter((p) => Object.keys(cell).some((k) => k.endsWith(`|${p}`)));
  if (!airlines.length) return empty("没有分组数据", "", "bar");
  const color = (rate) => `color-mix(in srgb, var(--success) ${Math.round(rate * 100)}%, var(--danger))`;
  return `<div class="heatmap" style="grid-template-columns: 96px repeat(${prods.length}, minmax(72px, 1fr))" role="table" aria-label="航司×票种正确率">
    <div class="hm-h" role="columnheader"></div>${prods.map((p) => `<div class="hm-h" role="columnheader">${esc(PRODUCTS[p] || "不限")}</div>`).join("")}
    ${airlines.map((a) => `<div class="hm-r" role="rowheader">${esc(a)} <span class="faint">${esc(AIRLINES[a] || "不限")}</span></div>${prods.map((p) => {
      const x = cell[`${a}|${p}`];
      if (!x) return `<div class="hm-c empty" role="cell" aria-label="无用例">·</div>`;
      const rate = x.ok / x.n;
      return `<div class="hm-c" role="cell" style="--hm:${color(rate)}" title="${a} × ${p}：${x.ok}/${x.n}"><span class="num">${Math.round(rate * 100)}%</span><span class="num faint">${x.ok}/${x.n}</span></div>`;
    }).join("")}`).join("")}</div>
    <div class="hm-legend"><span class="faint">0%</span><span class="hm-scale"></span><span class="faint">100%</span></div>`;
}

function reasonBars(fr) {
  const list = Array.isArray(fr) ? fr.map((x) => (Array.isArray(x) ? x : [x.reason ?? x.name, x.count ?? x.n])) : Object.entries(fr || {});
  const rows = list.filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
  if (!rows.length) return empty("没有失败", "全部用例通过", "checkCircle");
  const max = Math.max(...rows.map((r) => r[1]));
  return `<ul class="hbars">${rows.map(([k, n]) => `<li><span class="hb-label" title="${esc(k)}">${esc(k)}</span><span class="hb-track"><span style="width:${(n / max) * 100}%"></span></span><span class="num hb-val">${n}</span></li>`).join("")}</ul>`;
}
const show = (v) => (v === null || v === undefined || (Array.isArray(v) && !v.length) ? `<span class="faint">—</span>` : esc(Array.isArray(v) ? v.join(" → ") : typeof v === "object" ? JSON.stringify(v) : v));

/** 期望 vs 实际 逐字段比较 */
function compareRows(r) {
  const e = r.expected || {}, a = r.actual || {};
  const eFees = e.fees || {}, aFees = a.fees || {};
  const rows = [
    ["意图", e.intents, a.intents, JSON.stringify(e.intents || []) === JSON.stringify(a.intents || [])],
    ["工具序列", e.tools, a.tools, JSON.stringify(e.tools || []) === JSON.stringify(a.tools || [])],
    ...Object.keys(eFees).filter((k) => eFees[k] !== null && eFees[k] !== undefined).map((k) => [`金额 · ${INTENT[k] || k}`, eFees[k], aFees[k], eFees[k] === aFees[k]]),
    ["规则（须包含）", e.rule_ids, a.rule_ids, (e.rule_ids || []).every((x) => (a.rule_ids || []).includes(x))],
    ["结局", e.outcome, a.outcome, !e.outcome || e.outcome === a.outcome],
  ];
  if (e.schema_errors !== undefined || (a.schema_errors && a.schema_errors.length)) rows.push(["Schema 错误", e.schema_errors, a.schema_errors, JSON.stringify(e.schema_errors ?? []) === JSON.stringify(a.schema_errors ?? [])]);
  return rows;
}

export function openDrill(r, c) {
  const body = document.createElement("div");
  body.className = "stack";
  body.style.gap = "16px";
  body.innerHTML = `<div class="row">${r.passed ? `<span class="chip ok">${icon("check", "sm")}通过</span>` : `<span class="chip bad">${icon("x", "sm")}失败</span>`}
      ${(r.tags || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("")}<span class="spacer"></span><span class="num faint">${ms(r.ms)} · ${r.tokens ?? 0} tokens</span></div>
    ${c ? `<p><span class="faint">问题</span> ${esc(c.question)}</p>` : ""}
    ${(r.reasons || []).length ? `<div class="callout bad">${icon("xCircle")}<div><strong>失败原因</strong><ul class="issue-list" style="margin-top:4px">${r.reasons.map((x) => `<li class="bad">${icon("x", "sm")}<span>${esc(x)}</span></li>`).join("")}</ul></div></div>` : ""}
    <div class="table-wrap"><table class="table compare"><thead><tr><th>字段</th><th>期望</th><th>实际</th><th>结果</th></tr></thead><tbody>
      ${compareRows(r).map(([k, e, a, ok]) => `<tr class="${ok ? "" : "diff"}"><td>${esc(k)}</td><td class="num">${show(e)}</td><td class="num">${show(a)}</td><td>${ok ? `<span class="status ok">${icon("check", "sm")}一致</span>` : `<span class="status bad">${icon("x", "sm")}不一致</span>`}</td></tr>`).join("")}
    </tbody></table></div>
    ${r.actual?.text ? `<div class="mini-card"><div class="mini-label">${icon("chat", "sm")}实际答复</div><p style="white-space:pre-wrap">${esc(r.actual.text)}</p></div>` : ""}
    <div><h4 class="row" style="margin-bottom:8px">${icon("workflow", "sm")}执行轨迹</h4><div data-trace></div></div>`;
  renderTrace(body.querySelector("[data-trace]"), r.trace || []);
  drawer({ title: `${r.case_id} · ${r.title || ""}`, labelIcon: "target", width: "760px", body });
}

export function renderDashboard(root, run, cases) {
  const s = run.summary || {};
  const caseMap = Object.fromEntries((cases || []).map((c) => [c.id, c]));
  const results = run.results || [];
  const fails = results.filter((r) => !r.passed);
  root.innerHTML = `<div class="row between" style="margin:20px 0 12px"><h3 class="row">${icon("bar", "sm")}运行结果 <code>${esc(run.id)}</code></h3>
      <span class="row faint"><span class="chip">${run.mode === "deepseek" ? "DeepSeek" : "规则模式"}</span><span class="num">${esc((run.created_at || "").replace("T", " ").slice(0, 19))}</span></span></div>
    ${metricCards(s)}
    <div class="dash-grid">
      <section class="card"><div class="card-head"><h3>${icon("layers", "sm")}航司 × 票种 正确率</h3></div><div class="card-pad">${heatmap(results, caseMap)}</div></section>
      <section class="card"><div class="card-head"><h3>${icon("alert", "sm")}失败原因分布</h3></div><div class="card-pad">${reasonBars(s.failure_reasons)}</div></section>
    </div>
    <section class="card" style="margin-top:16px"><div class="card-head"><h3>${icon("xCircle", "sm")}失败用例 <span class="num muted">${fails.length}</span></h3><span class="faint">点击查看期望与实际对比和执行轨迹</span></div>
      ${fails.length ? `<div class="table-wrap flat"><table class="table"><thead><tr><th>ID</th><th>标题</th><th>失败原因</th><th>期望结局</th><th>实际结局</th><th class="num">耗时</th><th><span class="sr-only">操作</span></th></tr></thead><tbody>
        ${fails.map((r) => `<tr><td><code>${esc(r.case_id)}</code></td><td class="clip">${esc(r.title)}</td><td class="clip">${esc((r.reasons || []).join("；"))}</td><td>${outcomeChip(r.expected?.outcome)}</td><td>${outcomeChip(r.actual?.outcome)}</td><td class="num">${ms(r.ms)}</td>
          <td><button type="button" class="btn btn-sm" data-open="${esc(r.case_id)}">${icon("search", "sm")}下钻</button></td></tr>`).join("")}</tbody></table></div>` : empty("全部通过", "本次运行没有失败用例", "checkCircle")}
    </section>`;
  const open = (id) => { const r = results.find((x) => x.case_id === id); if (r) openDrill(r, caseMap[id]); else toastError(new Error("该用例不在当前运行结果中")); };
  root.querySelectorAll("[data-open]").forEach((b) => (b.onclick = () => open(b.dataset.open)));
  return { open };
}
/** 历史运行列表 + 两次运行对比 */
export function renderHistory(root, runs, { current, onOpen, onLoad }) {
  if (!runs.length) { root.innerHTML = empty("暂无历史运行", "", "history"); return; }
  const picked = new Set();
  root.innerHTML = `<div class="table-wrap flat"><table class="table"><thead><tr><th><span class="sr-only">对比</span></th><th>运行 ID</th><th>时间</th><th>模式</th><th class="num">用例</th><th class="num">正确率</th><th class="num">工具准确率</th><th class="num">政策命中</th><th class="num">转人工</th><th class="num">P95</th><th><span class="sr-only">操作</span></th></tr></thead><tbody>
    ${runs.map((r) => { const s = r.summary || {}; return `<tr class="${r.id === current ? "selected" : ""}">
      <td><input type="checkbox" data-pick="${esc(r.id)}" aria-label="选择 ${esc(r.id)} 用于对比"></td><td><code>${esc(r.id)}</code>${r.id === current ? ` <span class="chip info">当前</span>` : ""}</td>
      <td class="num faint">${esc((r.created_at || "").replace("T", " ").slice(0, 16))}</td><td>${r.mode === "deepseek" ? "DeepSeek" : "规则"}</td><td class="num">${s.n ?? "—"}</td>
      <td class="num">${pct(s.correctness)}</td><td class="num">${pct(s.tool_acc)}</td><td class="num">${pct(s.policy_hit)}</td><td class="num">${pct(s.human_rate)}</td><td class="num">${ms(s.latency_p95_ms)}</td>
      <td><button type="button" class="btn btn-sm btn-ghost" data-view="${esc(r.id)}">查看</button></td></tr>`; }).join("")}
    </tbody></table></div>
    <div class="card-pad row between"><span class="faint" data-hint>已选 0 / 2</span><button type="button" class="btn" data-compare disabled>${icon("git", "sm")}对比两次运行</button></div>
    <div data-compare-out></div>`;
  const btn = root.querySelector("[data-compare]");
  root.querySelectorAll("[data-pick]").forEach((x) => (x.onchange = () => {
    if (x.checked) { if (picked.size >= 2) { x.checked = false; return; } picked.add(x.dataset.pick); } else picked.delete(x.dataset.pick);
    root.querySelector("[data-hint]").textContent = `已选 ${picked.size} / 2`;
    btn.disabled = picked.size !== 2;
  }));
  root.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => onOpen(b.dataset.view)));
  btn.onclick = async () => {
    const out = root.querySelector("[data-compare-out]");
    try {
      const [a, b] = await Promise.all([...picked].map(onLoad));
      const [old, neu] = (a.created_at || "") <= (b.created_at || "") ? [a, b] : [b, a];
      const sa = old.summary || {}, sb = neu.summary || {};
      const ra = Object.fromEntries((old.results || []).map((r) => [r.case_id, r.passed]));
      const flips = (neu.results || []).filter((r) => r.case_id in ra && ra[r.case_id] !== r.passed);
      out.innerHTML = `<div class="card-pad stack" style="gap:12px"><h4 class="row">${icon("git", "sm")}<code>${esc(old.id)}</code> ${icon("arrowRight", "sm")} <code>${esc(neu.id)}</code></h4>
        <div class="table-wrap"><table class="table"><thead><tr><th>指标</th><th class="num">之前</th><th class="num">之后</th><th class="num">变化</th></tr></thead><tbody>
        ${METRICS.map(([k, label, f, up]) => {
          const x = sa[k], y = sb[k];
          if (x === undefined && y === undefined) return "";
          const d = (y ?? 0) - (x ?? 0);
          const better = d === 0 ? null : (d > 0) === up;
          const ds = f === "pct" ? `${d > 0 ? "+" : ""}${(d * 100).toFixed(1)} pp` : `${d > 0 ? "+" : ""}${f === "ms" ? ms(d) : Math.round(d)}`;
          return `<tr><td>${label}</td><td class="num">${fmt(x, f)}</td><td class="num">${fmt(y, f)}</td><td class="num">${better === null ? `<span class="faint">持平</span>` : `<span class="status ${better ? "ok" : "bad"}">${icon(better ? "check" : "alert", "sm")}${ds}</span>`}</td></tr>`;
        }).join("")}</tbody></table></div>
        ${flips.length ? `<div class="mini-card"><div class="mini-label">${icon("refresh", "sm")}结果翻转的用例</div><ul class="issue-list">${flips.map((r) => `<li class="${r.passed ? "" : "bad"}">${icon(r.passed ? "checkCircle" : "xCircle", "sm")}<span><code>${esc(r.case_id)}</code> ${r.passed ? "失败 → 通过" : "通过 → 失败"}</span></li>`).join("")}</ul></div>` : `<p class="faint">两次运行逐条结果一致</p>`}</div>`;
    } catch (e) { toastError(e, "对比失败"); }
  };
}
