// 配置中心：按模块分组的参数表单，字段说明 + 取值范围，保存显示行内错误，恢复默认，DeepSeek 测试连接
import { get, post, put } from "../api.js";
import { icon } from "../icons.js";
import { refreshStatus } from "../store.js";
import { esc, toast, toastError, confirmDialog, busy, skeleton } from "../ui.js";

// [path, 标签, 类型, min, max, step, 说明]
const GROUPS = [
  { id: "llm", title: "决策模式与 DeepSeek", icon: "brain", fields: [
    ["mode", "决策模式", "select:offline=规则模式,deepseek=DeepSeek", 0, 0, 0, "规则模式完全离线；DeepSeek 模式由大模型规划工具调用，出错时单步降级"],
    ["llm.base_url", "Base URL", "text", 0, 0, 0, "OpenAI 兼容接口地址"],
    ["llm.model", "模型", "text", 0, 0, 0, "如 deepseek-chat"],
    ["llm.api_key", "API Key", "password", 0, 0, 0, "只写不回显：显示为打码值，留空或不改动表示保持原值"],
    ["llm.timeout", "超时（秒）", "int", 1, 120, 1, "单次请求超时"],
  ] },
  { id: "query", title: "Query 理解", icon: "sparkles", fields: [
    ["query.clarify_threshold", "追问阈值", "num", 0, 1, 0.05, "解析置信度低于该值时向乘客追问"],
    ["query.step_back_min_hits", "Step-back 触发命中数", "int", 0, 10, 1, "检索命中少于该数时，做 Step-back 泛化 / HyDE 扩写再检索"],
  ] },
  { id: "retrieval", title: "检索漏斗", icon: "funnel", fields: [
    ["retrieval.backend", "检索后端", "select:hybrid=hybrid（向量+BM25+精排）,fallback=fallback（字符 n-gram）", 0, 0, 0, "hybrid 需要本地模型；加载失败会自动降级"],
    ["retrieval.k_recall", "召回 k_recall", "int", 1, 200, 1, "向量与 BM25 各召回条数"],
    ["retrieval.k_fuse", "融合 k_fuse", "int", 1, 200, 1, "RRF 融合后保留条数，≤ k_recall"],
    ["retrieval.k_rerank", "精排 k_rerank", "int", 1, 100, 1, "精排后保留条数，≤ k_fuse"],
    ["retrieval.k_mmr", "去冗余 k_mmr", "int", 1, 50, 1, "MMR 后保留条数，≤ k_rerank"],
    ["retrieval.k_inject", "注入 k_inject", "int", 1, 20, 1, "最终注入上下文条数，≤ k_mmr"],
    ["retrieval.rrf_k", "RRF 常数 rrf_k", "int", 1, 200, 1, "越大排名差异影响越平缓"],
    ["retrieval.rerank_min", "精排门槛 rerank_min", "num", 0, 1, 0.05, "归一化精排分低于该值的候选被截断（动态截断）"],
    ["retrieval.mmr_lambda", "MMR λ", "num", 0, 1, 0.05, "1 = 只看相关性，0 = 只看多样性"],
    ["retrieval.profile_boost", "画像加权", "num", 0, 1, 0.05, "常用航司/票种的规则得分乘以 (1+boost)"],
    ["retrieval.dense_min.hybrid", "向量门槛（hybrid）", "num", 0, 1, 0.05, "含领域词时向量相似度门槛"],
    ["retrieval.dense_min.fallback", "向量门槛（fallback）", "num", 0, 1, 0.05, "降级模式下的同一门槛"],
    ["retrieval.dense_strict.hybrid", "严格门槛（hybrid）", "num", 0, 1, 0.05, "无领域词时仍可入选的向量相似度"],
    ["retrieval.dense_strict.fallback", "严格门槛（fallback）", "num", 0, 1, 0.05, "降级模式下的同一门槛"],
  ] },
  { id: "memory", title: "记忆预算", icon: "layers", fields: [
    ["memory.token_budget", "Token 预算", "int", 200, 32000, 100, "拼接上下文的总预算（≥ 200，保证系统提示和订单不被截断）"],
    ["memory.recent_turns", "滑动窗口轮数", "int", 1, 20, 1, "超出的旧轮次压缩进摘要"],
    ["memory.order_stack", "订单栈深度", "int", 1, 20, 1, "用于「回程那张」之类的指代消解"],
    ["memory.budget_share.system", "份额 · 系统提示", "num", 0, 1, 0.05, ""],
    ["memory.budget_share.order", "份额 · 订单", "num", 0, 1, 0.05, ""],
    ["memory.budget_share.rules", "份额 · 规则", "num", 0, 1, 0.05, ""],
    ["memory.budget_share.history", "份额 · 历史", "num", 0, 1, 0.05, ""],
    ["memory.budget_share.profile", "份额 · 画像", "num", 0, 1, 0.05, "各块份额之和 ≤ 1；没用完的份额顺延给下一块"],
  ] },
  { id: "verify", title: "质量保障", icon: "shield", fields: [
    ["verify.threshold", "置信度阈值", "num", 0, 1, 0.05, "低于阈值不放行：先自我修正，仍不过则转人工"],
    ["verify.max_retries", "最大修正次数", "int", 0, 5, 1, "校验失败后重写方案的次数"],
  ] },
  { id: "planner", title: "Planner（ReAct）", icon: "workflow", fields: [
    ["planner.max_rounds", "最大轮数", "int", 1, 20, 1, "每个子诉求的 ReAct 轮数上限，超过转人工"],
    ["planner.max_tool_retries", "工具参数重试", "int", 0, 5, 1, "Schema 校验失败后回灌错误重试的次数"],
  ] },
];
const getPath = (o, p) => p.split(".").reduce((x, k) => (x == null ? x : x[k]), o);
const setPath = (o, p, v) => { const ks = p.split("."); let x = o; ks.slice(0, -1).forEach((k) => (x = x[k] ||= {})); x[ks[ks.length - 1]] = v; };
function control([path, , type, min, max, step], v) {
  const id = `cfg-${path}`;
  if (type.startsWith("select:")) {
    const opts = type.slice(7).split(",").map((x) => x.split("="));
    return `<select id="${id}" name="${path}">${opts.map(([k, l]) => `<option value="${k}" ${k === v ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
  }
  if (type === "text") return `<input class="input" id="${id}" name="${path}" value="${esc(v ?? "")}" spellcheck="false">`;
  if (type === "password") return `<div class="row" style="flex-wrap:nowrap"><input class="input mono" id="${id}" name="${path}" type="password" value="${esc(v ?? "")}" autocomplete="off" placeholder="sk-…" spellcheck="false">
    <button type="button" class="icon-btn" data-reveal aria-label="显示或隐藏 API Key">${icon("key", "sm")}</button></div>`;
  return `<input class="input num" id="${id}" name="${path}" type="number" inputmode="decimal" min="${min}" max="${max}" step="${step}" value="${v ?? ""}">`;
}

export async function mount(view) {
  view.innerHTML = `<div class="page config-page"><div class="page-head"><div><h2>配置中心</h2><p>运行参数实时生效；只保存与默认值不同的覆盖项到 config.json。</p></div>
    <div class="row"><span class="status idle" data-dirty></span><button class="btn" type="button" data-reset>${icon("rotate", "sm")}恢复默认</button><button class="btn btn-primary" type="button" data-save disabled>${icon("check", "sm")}保存</button></div></div>
    <div class="callout bad" data-summary hidden role="alert">${icon("xCircle")}<div><strong>保存失败</strong><ul class="issue-list" style="margin-top:4px"></ul></div></div>
    <form class="config-grid" data-form novalidate>${skeleton(6, 60)}</form></div>`;
  const $ = (s) => view.querySelector(s);
  const form = $("[data-form]");
  let cfg = null;

  const draw = () => {
    form.innerHTML = GROUPS.map((g) => `<fieldset class="card cfg-group" data-group="${g.id}"><legend>${icon(g.icon, "sm")}${esc(g.title)}</legend>
      ${g.id === "llm" ? `<div class="row llm-test"><span data-llm-status>${cfg.llm.api_key ? `<span class="status ok">${icon("key", "sm")}已配置 Key</span>` : `<span class="status warn">${icon("alert", "sm")}未配置 Key，DeepSeek 模式不可用</span>`}</span><span class="spacer"></span><button type="button" class="btn btn-sm" data-test>${icon("wifi", "sm")}测试连接</button></div>` : ""}
      ${g.id === "memory" ? `<div class="share-sum" data-share></div>` : ""}
      <div class="cfg-fields">${g.fields.map((f) => { const [path, label, type, min, max] = f; const rng = type === "int" || type === "num" ? `<span class="range">${min}–${max}</span>` : "";
        return `<div class="field" data-field="${path}"><label for="cfg-${path}">${esc(label)}${rng}</label>${control(f, getPath(cfg, path))}${f[6] ? `<span class="help">${esc(f[6])}</span>` : ""}<span class="err" hidden></span></div>`; }).join("")}</div></fieldset>`).join("");
    form.querySelector("[data-reveal]").onclick = () => { const i = form.querySelector('[name="llm.api_key"]'); i.type = i.type === "password" ? "text" : "password"; };
    form.querySelector("[data-test]").onclick = testLLM;
    shareSum(); dirty();
  };
  const read = () => {
    const out = {};
    GROUPS.flatMap((g) => g.fields).forEach(([path, , type]) => {
      const el = form.querySelector(`[name="${CSS.escape(path)}"]`);
      if (!el) return;
      const raw = el.value.trim();
      out[path] = type === "int" || type === "num" ? (raw === "" ? NaN : Number(raw)) : raw;
    });
    return out;
  };
  const shareSum = () => {
    const v = read();
    const s = ["system", "order", "rules", "history", "profile"].reduce((a, k) => a + (v[`memory.budget_share.${k}`] || 0), 0);
    const el = form.querySelector("[data-share]");
    if (el) el.innerHTML = s > 1.0001 ? `<span class="status bad">${icon("xCircle", "sm")}份额合计 <span class="num">${s.toFixed(2)}</span>，不能超过 1</span>` : `<span class="status ok">${icon("check", "sm")}份额合计 <span class="num">${s.toFixed(2)}</span></span>`;
  };
  const changes = () => {
    const v = read(), patch = {};
    let shareTouched = false;
    for (const [p, x] of Object.entries(v)) {
      const cur = getPath(cfg, p);
      if (p === "llm.api_key" && (x === "" || x === cur || x.includes("****"))) continue;
      if (x === cur || (typeof x === "number" && Number.isNaN(x) && cur == null)) continue;
      if (p.startsWith("memory.budget_share.")) { shareTouched = true; continue; }
      setPath(patch, p, x);
    }
    if (shareTouched) setPath(patch, "memory.budget_share", Object.fromEntries(["system", "order", "rules", "history", "profile"].map((k) => [k, v[`memory.budget_share.${k}`]])));
    return patch;
  };
  const dirty = () => {
    const n = JSON.stringify(changes()) !== "{}";
    $("[data-save]").disabled = !n;
    $("[data-dirty]").innerHTML = n ? `${icon("edit", "sm")}有未保存的修改` : "";
  };
  const showErrors = (fields) => {
    form.querySelectorAll(".field.invalid").forEach((x) => { x.classList.remove("invalid"); x.querySelector(".err").hidden = true; });
    const sum = $("[data-summary]");
    const entries = Object.entries(fields || {});
    sum.hidden = !entries.length;
    sum.querySelector("ul").innerHTML = entries.map(([k, m]) => `<li class="bad">${icon("xCircle", "sm")}<span><code>${esc(k)}</code> ${esc(m)}</span></li>`).join("");
    for (const [k, m] of entries) {
      const el = form.querySelector(`[data-field="${CSS.escape(k)}"]`) || form.querySelector(`[data-field^="${CSS.escape(k)}."]`);
      if (el) { el.classList.add("invalid"); const s = el.querySelector(".err"); s.hidden = false; s.innerHTML = `${icon("alert", "sm")}${esc(m)}`; }
    }
    form.querySelector(".field.invalid input, .field.invalid select")?.focus();
  };
  const localErrors = () => {
    const v = read(), errs = {};
    GROUPS.flatMap((g) => g.fields).forEach(([p, , type, min, max]) => {
      if (type !== "int" && type !== "num") return;
      const x = v[p];
      if (Number.isNaN(x)) errs[p] = "必填";
      else if (type === "int" && !Number.isInteger(x)) errs[p] = "必须是整数";
      else if (x < min || x > max) errs[p] = `取值范围 ${min}–${max}`;
    });
    return errs;
  };
  const save = async () => {
    const le = localErrors();
    if (Object.keys(le).length) { showErrors(le); return; }
    const patch = changes();
    if (!Object.keys(patch).length) return;
    const done = busy($("[data-save]"), "保存中");
    try {
      cfg = await put("/api/config", patch);
      showErrors({}); draw(); toast("配置已保存并生效", "ok"); refreshStatus().catch(() => {});
    } catch (e) { showErrors(e.fields); toastError(e, "保存失败"); }
    finally { done(); dirty(); }
  };
  const testLLM = async (e) => {
    const btn = e.currentTarget, done = busy(btn, "测试中");
    const st = form.querySelector("[data-llm-status]");
    if (Object.keys(changes().llm || {}).length) toast("测试使用已保存的配置，请先保存 DeepSeek 修改", "warn");
    try {
      const r = await post("/api/config/test-llm");
      st.innerHTML = r.ok ? `<span class="status ok">${icon("checkCircle", "sm")}连接成功 · ${esc(r.model)} · <span class="num">${r.ms}ms</span></span>` : `<span class="status bad">${icon("xCircle", "sm")}${esc(r.error)}</span>`;
    } catch (err) { toastError(err, "测试失败"); } finally { done(); }
  };

  form.addEventListener("input", () => { shareSum(); dirty(); });
  form.addEventListener("change", () => { shareSum(); dirty(); });
  form.addEventListener("submit", (e) => { e.preventDefault(); save(); });
  $("[data-save]").onclick = save;
  $("[data-reset]").onclick = async () => {
    if (!(await confirmDialog({ title: "恢复默认", message: "清除所有覆盖项，恢复默认参数（保留 DeepSeek API Key）。", okText: "恢复默认", danger: true }))) return;
    try { cfg = await post("/api/config/reset"); showErrors({}); draw(); toast("已恢复默认配置", "ok"); refreshStatus().catch(() => {}); } catch (e) { toastError(e); }
  };
  try { cfg = await get("/api/config"); draw(); } catch (e) { toastError(e, "配置加载失败"); }
}
