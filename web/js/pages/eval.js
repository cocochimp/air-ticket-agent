// 评测中心：用例表（筛选/搜索/勾选）、编辑抽屉、导入导出、运行面板（SSE 进度）、结果看板、历史对比
import { get, post, put, stream } from "../api.js";
import { icon } from "../icons.js";
import { state } from "../store.js";
import { esc, empty, skeleton, toast, toastError, confirmDialog, downloadJSON, pickJSON, busy, debounce, outcomeChip, AIRLINES, PRODUCTS, INTENT } from "../ui.js";
import { openCaseEditor } from "./eval-form.js";
import { renderDashboard, renderHistory } from "./eval-results.js";

const BLANK = { id: "", title: "", tags: [], user_id: "u_zhang", airline: null, fare_product: null, order: "#1", question: "", history: [], clarify_answers: [], expect: { intents: ["CHANGE"], tools: ["tool_flight_status", "tool_rule_retrieve", "tool_recalc_fee"], fees: {}, rule_ids: [], outcome: "ANSWER", schema_errors: [] }, forbid: ["quote_mismatch", "unsupported_claim", "skip_rule"] };

export async function mount(view) {
  view.innerHTML = `<div class="page eval-page">
    <div class="page-head"><div><h2>评测中心</h2><p>金标准用例管理、批量运行与结果分析。规则模式下应保持方案正确率 ≥ 90%。</p></div>
      <div class="row">
        <button class="btn" type="button" data-import>${icon("upload", "sm")}导入</button>
        <button class="btn" type="button" data-export>${icon("download", "sm")}导出</button>
        <button class="btn btn-primary" type="button" data-add>${icon("plus", "sm")}新建用例</button>
      </div></div>
    <div class="eval-top">
      <section class="card" aria-labelledby="cases-h">
        <div class="card-head"><h3 id="cases-h">${icon("flask", "sm")}用例 <span class="num muted" data-count></span></h3>
          <div class="row filters">
            <div class="search"><span class="sr-only">搜索</span>${icon("search", "sm")}<input class="input" type="search" placeholder="搜索 ID / 标题 / 问题" data-q aria-label="搜索用例"></div>
            <select class="input" data-tag aria-label="按标签筛选"><option value="">全部标签</option></select>
            <select class="input" data-airline aria-label="按航司筛选"><option value="">全部航司</option>${Object.entries(AIRLINES).map(([k, v]) => `<option value="${k}">${k} ${v}</option>`).join("")}</select>
            <select class="input" data-outcome aria-label="按结局筛选"><option value="">全部结局</option><option value="ANSWER">已答复</option><option value="HUMAN">转人工</option><option value="CLARIFY">追问</option></select>
          </div></div>
        <div class="table-wrap flat" data-table>${skeleton(6, 28)}</div>
      </section>
      <section class="card run-card" aria-labelledby="run-h">
        <div class="card-head"><h3 id="run-h">${icon("play", "sm")}运行评测</h3></div>
        <div class="card-pad stack" style="gap:14px">
          <div class="field"><span class="field-label">范围</span><div class="segmented" role="radiogroup" aria-label="运行范围" data-scope>
            <button type="button" role="radio" aria-checked="true" data-v="all">全部</button><button type="button" role="radio" aria-checked="false" data-v="filtered">筛选结果</button><button type="button" role="radio" aria-checked="false" data-v="selected">已勾选</button></div>
            <span class="help num" data-scope-n></span></div>
          <div class="field"><span class="field-label">模式</span><div class="segmented" role="radiogroup" aria-label="评测模式" data-mode>
            <button type="button" role="radio" aria-checked="true" data-v="offline">${icon("shield", "sm")}规则模式</button><button type="button" role="radio" aria-checked="false" data-v="deepseek">${icon("brain", "sm")}DeepSeek</button></div></div>
          <button class="btn btn-primary btn-lg" type="button" data-run>${icon("play", "sm")}开始运行</button>
          <div class="run-progress" data-progress hidden>
            <div class="row between"><span class="status run" data-pstatus>${icon("loader", "sm spin")}运行中</span><span class="num muted" data-pnum>0 / 0</span></div>
            <div class="progress" role="progressbar" aria-label="评测进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><span style="width:0%"></span></div>
            <div class="run-log" data-plog aria-live="polite"></div>
          </div>
        </div>
      </section>
    </div>
    <section class="eval-results" data-results aria-label="评测结果"></section>
    <section class="card" aria-labelledby="hist-h" style="margin-top:16px"><div class="card-head"><h3 id="hist-h">${icon("history", "sm")}历史运行</h3><span class="faint">勾选两次运行进行对比</span></div><div data-history>${skeleton(3, 24)}</div></section>
  </div>`;
  const $ = (s) => view.querySelector(s);
  let cases = [], users = [], selected = new Set(), scope = "all", mode = "offline", lastRun = null, ctrl = null, alive = true;
  const F = { q: "", tag: "", airline: "", outcome: "" };

  const filtered = () => cases.filter((c) => {
    if (F.tag && !(c.tags || []).includes(F.tag)) return false;
    if (F.airline && c.airline !== F.airline) return false;
    if (F.outcome && (c.expect?.outcome || "ANSWER") !== F.outcome) return false;
    if (F.q) { const q = F.q.toLowerCase(); return [c.id, c.title, c.question].some((x) => String(x || "").toLowerCase().includes(q)); }
    return true;
  });
  const scopeIds = () => (scope === "all" ? null : scope === "filtered" ? filtered().map((c) => c.id) : [...selected]);
  const lastResult = (id) => lastRun?.results?.find((r) => r.case_id === id);
  const userName = (id) => users.find((u) => u.user_id === id)?.name || id;
  const feeText = (fees = {}) => Object.entries(fees).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => `${INTENT[k] || k} <span class="num">¥${v}</span>`).join("<br>") || `<span class="faint">—</span>`;

  const drawTable = () => {
    const rows = filtered();
    $("[data-count]").textContent = `${rows.length} / ${cases.length}`;
    const n = scopeIds();
    $("[data-scope-n]").textContent = `将运行 ${n ? n.length : cases.length} 条`;
    if (!cases.length) { $("[data-table]").innerHTML = empty("还没有用例", "eval/gold_cases.json 不存在或为空；可以新建或导入", "flask"); return; }
    if (!rows.length) { $("[data-table]").innerHTML = empty("没有符合筛选条件的用例", "调整搜索词或筛选条件", "search"); return; }
    const allOn = rows.every((c) => selected.has(c.id));
    $("[data-table]").innerHTML = `<table class="table"><thead><tr>
      <th><input type="checkbox" aria-label="全选当前筛选结果" data-all ${allOn ? "checked" : ""}></th><th>ID</th><th>标题 / 问题</th><th>标签</th><th>用户</th><th>航司·票种</th><th>期望金额</th><th>结局</th><th>上次</th><th><span class="sr-only">操作</span></th></tr></thead>
      <tbody>${rows.map((c) => { const r = lastResult(c.id); return `<tr class="${selected.has(c.id) ? "selected" : ""}">
        <td><input type="checkbox" data-sel="${esc(c.id)}" aria-label="选择 ${esc(c.id)}" ${selected.has(c.id) ? "checked" : ""}></td>
        <td><code>${esc(c.id)}</code></td>
        <td class="clip"><div>${esc(c.title)}</div><div class="faint clip">${esc(c.question)}</div></td>
        <td>${(c.tags || []).map((t) => `<span class="chip">${esc(t)}</span>`).join(" ")}</td>
        <td>${esc(userName(c.user_id))}</td>
        <td>${esc(c.airline || "—")} <span class="faint">${esc(PRODUCTS[c.fare_product] || c.fare_product || "")}</span></td>
        <td>${feeText(c.expect?.fees)}</td>
        <td>${outcomeChip(c.expect?.outcome || "ANSWER")}</td>
        <td>${r ? (r.passed ? `<span class="status ok">${icon("check", "sm")}通过</span>` : `<button type="button" class="btn btn-ghost btn-sm" data-drill="${esc(c.id)}"><span class="status bad">${icon("x", "sm")}失败</span></button>`) : `<span class="faint">—</span>`}</td>
        <td class="row-actions"><button type="button" class="icon-btn sm" data-edit="${esc(c.id)}" aria-label="编辑 ${esc(c.id)}">${icon("edit", "sm")}</button><button type="button" class="icon-btn sm" data-dup="${esc(c.id)}" aria-label="复制 ${esc(c.id)}">${icon("copy", "sm")}</button><button type="button" class="icon-btn sm" data-del="${esc(c.id)}" aria-label="删除 ${esc(c.id)}">${icon("trash", "sm")}</button></td>
      </tr>`; }).join("")}</tbody></table>`;
    const T = $("[data-table]");
    T.querySelector("[data-all]").onchange = (e) => { rows.forEach((c) => (e.target.checked ? selected.add(c.id) : selected.delete(c.id))); drawTable(); };
    T.querySelectorAll("[data-sel]").forEach((x) => (x.onchange = () => { x.checked ? selected.add(x.dataset.sel) : selected.delete(x.dataset.sel); drawTable(); }));
    T.querySelectorAll("[data-edit]").forEach((b) => (b.onclick = () => edit(cases.find((c) => c.id === b.dataset.edit), false)));
    T.querySelectorAll("[data-dup]").forEach((b) => (b.onclick = () => { const c = cases.find((x) => x.id === b.dataset.dup); edit({ ...structuredClone(c), id: `${c.id}_copy` }, true); }));
    T.querySelectorAll("[data-del]").forEach((b) => (b.onclick = () => remove(b.dataset.del)));
    T.querySelectorAll("[data-drill]").forEach((b) => (b.onclick = () => dash?.open(b.dataset.drill)));
  };
  const fillTags = () => {
    const tags = [...new Set(cases.flatMap((c) => c.tags || []))].sort();
    $("[data-tag]").innerHTML = `<option value="">全部标签</option>${tags.map((t) => `<option ${t === F.tag ? "selected" : ""}>${esc(t)}</option>`).join("")}`;
  };
  const loadCases = async () => {
    try { cases = (await get("/api/eval/cases")).cases || []; }
    catch (e) { cases = []; toastError(e, "用例加载失败"); }
    selected = new Set([...selected].filter((id) => cases.some((c) => c.id === id)));
    fillTags(); drawTable();
  };
  const saveAll = async (next, msg) => {
    try { await put("/api/eval/cases", { cases: next }); cases = next; fillTags(); drawTable(); toast(msg, "ok"); }
    catch (e) {
      const f = e.fields?.cases;
      toastError(f ? new Error(Object.entries(f).map(([id, es]) => `${id}: ${es.map((x) => `${x.field} ${x.message}`).join("；")}`).join("\n")) : e, "保存失败");
      throw e;
    }
  };
  const edit = (c, isNew) => openCaseEditor({
    c: structuredClone(c), users, isNew, existingIds: new Set(cases.map((x) => x.id)),
    onSave: (nc, oldId) => saveAll(isNew ? [...cases, nc] : cases.map((x) => (x.id === oldId ? nc : x)), isNew ? `已新建 ${nc.id}` : `已保存 ${nc.id}`),
  });
  const remove = async (id) => {
    if (await confirmDialog({ title: "删除用例", message: `确定删除用例 ${id}？保存前会自动备份为 gold_cases.json.bak。`, okText: "删除", danger: true })) {
      selected.delete(id);
      saveAll(cases.filter((c) => c.id !== id), `已删除 ${id}`).catch(() => {});
    }
  };

  // ---------- 工具栏 ----------
  $("[data-add]").onclick = () => edit({ ...structuredClone(BLANK), id: `case_${String(cases.length + 1).padStart(3, "0")}`, user_id: state.userId }, true);
  $("[data-export]").onclick = () => downloadJSON(cases, `gold_cases_${new Date().toISOString().slice(0, 10)}.json`);
  $("[data-import]").onclick = async () => {
    try {
      const data = await pickJSON();
      const list = Array.isArray(data) ? data : data.cases;
      if (!Array.isArray(list)) throw new Error("JSON 必须是用例数组或 {cases: [...]}");
      const r = await post("/api/eval/cases/validate", { cases: list });
      if (!r.ok) {
        const n = Object.keys(r.errors).length;
        toast(`${n} 条用例未通过校验：${Object.entries(r.errors).slice(0, 3).map(([id, es]) => `${id}（${es[0]?.field} ${es[0]?.message}）`).join("；")}`, "bad", "导入失败");
        return;
      }
      if (await confirmDialog({ title: "导入用例", message: `将用导入的 ${list.length} 条用例替换当前 ${cases.length} 条（自动备份）。继续？`, okText: "替换" })) await saveAll(list, `已导入 ${list.length} 条用例`);
    } catch (e) { if (e) toastError(e, "导入失败"); }
  };
  const debounced = debounce(() => drawTable(), 150);
  $("[data-q]").oninput = (e) => { F.q = e.target.value.trim(); debounced(); };
  $("[data-tag]").onchange = (e) => { F.tag = e.target.value; drawTable(); };
  $("[data-airline]").onchange = (e) => { F.airline = e.target.value; drawTable(); };
  $("[data-outcome]").onchange = (e) => { F.outcome = e.target.value; drawTable(); };
  const radio = (sel, set) => view.querySelectorAll(`${sel} [data-v]`).forEach((b) => (b.onclick = () => {
    view.querySelectorAll(`${sel} [data-v]`).forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    set(b.dataset.v); drawTable();
  }));
  radio("[data-scope]", (v) => (scope = v));
  radio("[data-mode]", (v) => { mode = v; if (v === "deepseek" && !state.status?.deepseek_configured) toast("未配置 DeepSeek API Key，运行将被拒绝", "warn"); });
  // ---------- 运行 ----------
  const prog = $("[data-progress]"), bar = prog.querySelector(".progress"), runBtn = $("[data-run]");
  const setProg = (done, total) => {
    const p = total ? Math.round((done / total) * 100) : 0;
    bar.firstElementChild.style.width = `${p}%`;
    bar.setAttribute("aria-valuenow", String(p));
    $("[data-pnum]").textContent = `${done} / ${total || "?"}`;
  };
  const watch = async (runId, total) => {
    prog.hidden = false; runBtn.disabled = true;
    $("[data-pstatus]").className = "status run";
    $("[data-pstatus]").innerHTML = `${icon("loader", "sm spin")}运行中 <code>${esc(runId)}</code>`;
    $("[data-plog]").innerHTML = ""; setProg(0, total);
    let fails = 0;
    ctrl = new AbortController();
    await stream("GET", `/api/eval/runs/${encodeURIComponent(runId)}/stream`, null, {
      progress: (m) => {
        setProg(m.done, m.total);
        if (!m.passed) { fails++; $("[data-plog]").insertAdjacentHTML("afterbegin", `<div class="status bad">${icon("x", "sm")}<code>${esc(m.case_id)}</code> 未通过</div>`); }
      },
      done: async (m) => {
        $("[data-pstatus]").className = "status ok";
        $("[data-pstatus]").innerHTML = `${icon("checkCircle", "sm")}已完成 <code>${esc(runId)}</code>`;
        toast(`评测完成：正确率 ${((m.summary?.correctness ?? 0) * 100).toFixed(1)}%，失败 ${fails} 条`, "ok");
        await showRun(runId); loadHistory();
      },
      error: (e) => {
        $("[data-pstatus]").className = "status bad";
        $("[data-pstatus]").innerHTML = `${icon("xCircle", "sm")}运行失败`;
        $("[data-plog]").insertAdjacentHTML("afterbegin", `<pre class="json">${esc(e.message)}\n${esc(e.stderr || "")}</pre>`);
        toast(e.message, "bad", "评测运行失败");
      },
    }, ctrl.signal);
    if (alive) runBtn.disabled = false;
  };
  runBtn.onclick = async () => {
    const ids = scopeIds();
    if (ids && !ids.length) { toast(scope === "selected" ? "请先勾选要运行的用例" : "筛选结果为空", "warn"); return; }
    const done = busy(runBtn, "启动中");
    try { const r = await post("/api/eval/runs", { mode, ids }); done(); watch(r.run_id, ids ? ids.length : cases.length); }
    catch (e) { done(); toastError(e, e.code === "RUN_ACTIVE" ? "已有评测在运行" : "无法启动评测"); }
  };

  // ---------- 结果与历史 ----------
  let dash = null;
  const showRun = async (id) => {
    try { lastRun = await get(`/api/eval/runs/${encodeURIComponent(id)}`); }
    catch (e) { toastError(e, "运行结果加载失败"); return; }
    dash = renderDashboard($("[data-results]"), lastRun, cases);
    drawTable();
  };
  const loadHistory = async () => {
    try {
      const { runs, active } = await get("/api/eval/runs");
      renderHistory($("[data-history]"), runs, { current: lastRun?.id, onOpen: showRun, onLoad: (id) => get(`/api/eval/runs/${encodeURIComponent(id)}`) });
      return { runs, active };
    } catch (e) { $("[data-history]").innerHTML = empty("历史加载失败", e.message, "alert"); return { runs: [], active: null }; }
  };

  try { users = (await get("/api/users")).users; } catch { users = state.users; }
  await loadCases();
  const { runs, active } = await loadHistory();
  if (runs[0]) await showRun(runs[0].id);
  else $("[data-results]").innerHTML = `<div class="card">${empty("还没有运行记录", "选择范围和模式后点击「开始运行」", "bar")}</div>`;
  if (active) watch(active, null);
  return () => { alive = false; ctrl?.abort(); };
}
