// 对话页右侧执行面板：流程 / 检索 / 记忆 / 校验
import { icon } from "../icons.js";
import { renderTrace } from "../trace.js";
import { renderFunnel } from "../funnel.js";
import { esc, empty, money, tabs, INTENT } from "../ui.js";

export function createPanel(root, { threshold = 0.85, maxRetries = 1 } = {}) {
  root.innerHTML = `<div class="panel-tabs"></div>
    <div class="panel-body">
      <section data-pane="trace" class="pane"></section>
      <section data-pane="retrieval" class="pane" hidden></section>
      <section data-pane="memory" class="pane" hidden></section>
      <section data-pane="verify" class="pane" hidden></section>
    </div>`;
  const panes = Object.fromEntries([...root.querySelectorAll("[data-pane]")].map((p) => [p.dataset.pane, p]));
  let events = [], pending = null, result = null, memory = null, sub = 0, funnelApi = null, hl = null;
  const t = tabs(root.querySelector(".panel-tabs"), [
    { id: "trace", label: "流程", icon: "workflow" }, { id: "retrieval", label: "检索", icon: "funnel" },
    { id: "memory", label: "记忆", icon: "brain" }, { id: "verify", label: "校验", icon: "gauge" },
  ], (id) => { Object.entries(panes).forEach(([k, p]) => (p.hidden = k !== id)); });
  Object.entries(panes).forEach(([k, p]) => { p.setAttribute("role", "tabpanel"); p.setAttribute("aria-label", k); });

  const drawTrace = () => renderTrace(panes.trace, events, { pending });
  const drawRetrieval = () => {
    const rs = (result?.results || []).filter((r) => r.funnel && r.funnel.length);
    if (!rs.length) { panes.retrieval.innerHTML = empty("暂无检索数据", "完成一次包含规则检索的对话后显示漏斗", "funnel"); funnelApi = null; return; }
    sub = Math.min(sub, rs.length - 1);
    const r = rs[sub];
    const pick = rs.length > 1 ? `<div class="segmented" role="group" aria-label="子诉求">${rs.map((x, i) => `<button type="button" aria-pressed="${i === sub}" data-sub="${i}">${esc(INTENT[x.intent] || x.intent)}</button>`).join("")}</div>` : "";
    const ret = r.retrieval || {};
    const meta = `<div class="pane-meta row between">${pick}<span class="row">
      ${ret.degraded ? `<span class="chip warn">${icon("alert", "sm")}降级 fallback</span>` : `<span class="chip ok">${icon("layers", "sm")}${esc(ret.backend || "hybrid")}</span>`}
      ${r.rewrite ? `<span class="chip info" title="${esc(r.rewrite.query)}">${icon("rotate", "sm")}${esc(r.rewrite.type)}</span>` : ""}</span></div>
      <p class="pane-q"><span class="faint">子诉求</span> ${esc(r.text)}</p>`;
    funnelApi = renderFunnel(panes.retrieval, r.funnel, { meta, highlight: hl });
    panes.retrieval.querySelectorAll("[data-sub]").forEach((b) => (b.onclick = () => { sub = +b.dataset.sub; hl = null; drawRetrieval(); }));
  };
  const BLOCKS = [["system", "系统提示"], ["order", "订单"], ["rules", "规则"], ["history", "历史"], ["profile", "画像"]];
  const drawMemory = () => {
    const ctx = result?.context, mem = result?.memory || memory;
    if (!ctx && !mem) { panes.memory.innerHTML = empty("暂无记忆数据", "对话后显示订单栈、摘要与 Token 预算", "brain"); return; }
    let html = "";
    if (mem) {
      html += `<div class="mem-grid">
        <div class="mini-card"><div class="mini-label">${icon("layers", "sm")}订单栈</div>${(mem.order_stack || []).length ? `<ol class="stack-list">${mem.order_stack.map((o, i) => `<li><span class="num faint">${i + 1}</span><code>${esc(o)}</code>${i === 0 ? `<span class="chip info">当前</span>` : ""}</li>`).join("")}</ol>` : `<p class="faint">空</p>`}</div>
        <div class="mini-card"><div class="mini-label">${icon("history", "sm")}会话</div><dl class="kv"><dt>窗口轮数</dt><dd class="num">${Array.isArray(mem.turns) ? mem.turns.length : mem.turns ?? 0}</dd><dt>上一意图</dt><dd>${esc(INTENT[mem.last_intent] || mem.last_intent || "—")}</dd></dl></div>
      </div>
      <div class="mini-card"><div class="mini-label">${icon("book", "sm")}滑动窗口摘要</div><p class="${mem.summary ? "" : "faint"}">${esc(mem.summary || "尚未产生摘要（超过窗口的旧轮次会被压缩到这里）")}</p></div>`;
    }
    if (ctx && ctx.blocks) {
      const full = Math.max(ctx.full_total || 0, ctx.total || 0, ctx.budget || 0, 1);
      const seg = (b, i) => { const v = ctx.blocks[b]?.tokens || 0; return v ? `<span class="tb-seg c${i}" style="width:${(v / full) * 100}%" title="${BLOCKS[i][1]} ${v} tokens"></span>` : ""; };
      const prof = ctx.blocks.profile?.text;
      html += `<div class="mini-card"><div class="row between"><div class="mini-label">${icon("bar", "sm")}Token 预算</div><span class="saved"><span class="big num">${ctx.saved_pct ?? 0}%</span><span class="muted">较全量节省</span></span></div>
        <div class="tb"><div class="tb-row"><span class="tb-name">预算拼接</span><div class="tb-track">${BLOCKS.map(([b], i) => seg(b, i)).join("")}<span class="tb-budget" style="left:${(ctx.budget / full) * 100}%" title="预算 ${ctx.budget}"></span></div><span class="num tb-val">${ctx.total}</span></div>
        <div class="tb-row"><span class="tb-name">全量拼接</span><div class="tb-track"><span class="tb-seg full" style="width:${(ctx.full_total / full) * 100}%"></span><span class="tb-budget" style="left:${(ctx.budget / full) * 100}%"></span></div><span class="num tb-val">${ctx.full_total}</span></div></div>
        <ul class="legend">${BLOCKS.map(([, l], i) => `<li><span class="sw c${i}"></span>${l}</li>`).join("")}<li><span class="sw budget"></span>预算 ${ctx.budget}</li></ul>
        <table class="table compact"><thead><tr><th>块</th><th class="num">已用</th><th class="num">配额</th><th>截断</th></tr></thead><tbody>${BLOCKS.map(([b, l]) => { const x = ctx.blocks[b] || {}; return `<tr><td>${l}</td><td class="num">${x.tokens ?? 0}</td><td class="num">${x.budget ?? 0}</td><td>${x.truncated ? `<span class="status warn">${icon("alert", "sm")}已截断</span>` : `<span class="status ok">${icon("check", "sm")}完整</span>`}</td></tr>`; }).join("")}</tbody></table>
      </div>
      ${prof ? `<div class="mini-card"><div class="mini-label">${icon("user", "sm")}长期画像（注入文本）</div><p>${esc(prof)}</p></div>` : ""}`;
    }
    panes.memory.innerHTML = html;
  };
  const drawVerify = () => {
    const v = result?.verify && Object.keys(result.verify).length ? result.verify : result?.final?.verify;
    if (!v || v.confidence === undefined) { panes.verify.innerHTML = empty("暂无校验结果", "方案生成后会比对金额、核查论断依据", "gauge"); return; }
    const retries = events.filter((e) => e.node === "retry_node").length;
    const rows = [];
    (result.final?.options || []).forEach((o, i) => {
      const fee = result.results?.[i]?.fee;
      if (!fee || !fee.allowed || !["CHANGE", "REFUND"].includes(o.type)) return;
      const eng = { "改期费": fee.change_fee, "票面差价": fee.fare_diff, "退票费": fee.refund_fee, "机场建设费退还": fee.tax };
      (o.fee_items || []).forEach((f) => rows.push([`${INTENT[o.type]} · ${f.name}`, f.amount, eng[f.name]]));
      rows.push([`${INTENT[o.type]} · 合计`, o.total, o.type === "CHANGE" ? fee.total : fee.refund_amount]);
    });
    const list = (arr, ic, cls) => arr.map((x) => `<li class="${cls}">${icon(ic, "sm")}<span>${esc(typeof x === "string" ? x : JSON.stringify(x))}</span></li>`).join("");
    panes.verify.innerHTML = `<div class="verify-top">
        ${gauge(v.confidence, threshold)}
        <div class="stack">
          ${v.passed ? `<span class="status ok big-status">${icon("checkCircle")}校验通过</span>` : `<span class="status bad big-status">${icon("xCircle")}校验未通过</span>`}
          <dl class="kv"><dt>置信度</dt><dd class="num">${(v.confidence ?? 0).toFixed(2)}</dd><dt>阈值</dt><dd class="num">${threshold}</dd>
          <dt>论断依据率</dt><dd class="num">${v.support_ratio !== undefined ? Math.round(v.support_ratio * 100) + "%" : "—"}</dd>
          <dt>自我修正</dt><dd class="num">${retries} / ${maxRetries} 次</dd>
          <dt>金额一致</dt><dd>${v.quote_mismatch ? `<span class="status bad">${icon("x", "sm")}不一致</span>` : `<span class="status ok">${icon("check", "sm")}一致</span>`}</dd></dl>
        </div></div>
      <div class="mini-card"><div class="mini-label">${icon("bar", "sm")}金额比对（方案 vs 运价引擎）</div>
        ${rows.length ? `<table class="table compact"><thead><tr><th>项目</th><th class="num">方案</th><th class="num">引擎</th><th>结果</th></tr></thead><tbody>${rows.map(([n, a, b]) => `<tr><td>${esc(n)}</td><td class="num">${money(a)}</td><td class="num">${money(b)}</td><td>${a === b ? `<span class="status ok">${icon("check", "sm")}一致</span>` : `<span class="status bad">${icon("x", "sm")}不符</span>`}</td></tr>`).join("")}</tbody></table>` : `<p class="faint">本轮没有需要比对的金额</p>`}</div>
      <div class="mini-card"><div class="mini-label">${icon("shield", "sm")}论断依据</div>
        ${(v.unsupported_claims || []).length ? `<ul class="issue-list">${list(v.unsupported_claims, "alert", "warn")}</ul>` : `<p class="status ok">${icon("check", "sm")}所有论断都能在规则原文中找到依据</p>`}
        ${(v.bad_refs || []).length ? `<p class="muted" style="margin-top:8px">无效引用</p><ul class="issue-list">${list(v.bad_refs, "xCircle", "bad")}</ul>` : ""}</div>
      ${(v.errors || []).length ? `<div class="mini-card"><div class="mini-label">${icon("alert", "sm")}校验问题</div><ul class="issue-list">${list(v.errors, "xCircle", "bad")}</ul></div>` : ""}`;
  };

  const all = () => { drawTrace(); drawRetrieval(); drawMemory(); drawVerify(); };
  all();
  return {
    reset() { events = []; pending = null; result = null; hl = null; sub = 0; all(); t.badge("trace", ""); },
    start(hint) { events = []; pending = hint; drawTrace(); t.badge("trace", `<span class="tab-dot" aria-hidden="true"></span>`); },
    push(e, hint) { events.push(e); pending = hint; drawTrace(); },
    hint(h) { pending = h; drawTrace(); },
    done(res) {
      pending = null;
      if (res) { result = res; if (res.trace && res.trace.length >= events.length) events = res.trace; sub = 0; hl = null; }
      all(); t.badge("trace", "");
    },
    setMemory(m) { memory = m; drawMemory(); },
    showRule(id) {
      hl = id;
      const rs = (result?.results || []).filter((r) => r.funnel && r.funnel.length);
      const i = rs.findIndex((r) => r.funnel.some((s) => s.items.some((x) => x.id === id)));
      if (i >= 0) sub = i;
      t.select("retrieval");
      drawRetrieval();
    },
    select: (id) => t.select(id),
  };
}

/** 半圆仪表：value 0-1，阈值刻度 */
export function gauge(value, threshold) {
  const v = Math.max(0, Math.min(1, value || 0));
  const R = 52, C = Math.PI * R;
  const ang = Math.PI * (1 - threshold);
  const tx = 64 + R * Math.cos(ang), ty = 64 - R * Math.sin(ang);
  const tx2 = 64 + (R + 12) * Math.cos(ang), ty2 = 64 - (R + 12) * Math.sin(ang);
  const cls = v >= threshold ? "ok" : v >= 0.5 ? "warn" : "bad";
  return `<figure class="gauge ${cls}" aria-label="置信度 ${v.toFixed(2)}，阈值 ${threshold}">
    <svg viewBox="0 0 128 76" role="img" aria-hidden="true">
      <path d="M12 64 A52 52 0 0 1 116 64" class="g-track"/>
      <path d="M12 64 A52 52 0 0 1 116 64" class="g-val" stroke-dasharray="${C * v} ${C}"/>
      <line x1="${tx.toFixed(1)}" y1="${ty.toFixed(1)}" x2="${tx2.toFixed(1)}" y2="${ty2.toFixed(1)}" class="g-th"/>
    </svg>
    <figcaption><span class="num big">${v.toFixed(2)}</span><span class="faint">阈值 ${threshold}</span></figcaption></figure>`;
}
