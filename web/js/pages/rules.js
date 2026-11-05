// 规则库：规则表（筛选）、编辑抽屉、检索调试器（复用漏斗渲染）
import { get, post, put, del } from "../api.js";
import { icon } from "../icons.js";
import { renderFunnel } from "../funnel.js";
import { state } from "../store.js";
import { esc, empty, skeleton, drawer, busy, toast, toastError, confirmDialog, debounce, ms, AIRLINES, PRODUCTS, SCENARIOS } from "../ui.js";

const CABINS = { F: "头等舱", C: "公务舱", Y: "经济舱" };
const opts = (map, cur, blank) => `${blank ? `<option value="">${blank}</option>` : ""}${Object.entries(map).map(([k, v]) => `<option value="${k}" ${k === cur ? "selected" : ""}>${k} · ${v}</option>`).join("")}`;
const withAny = (map) => ({ "*": "通用", ...map });

export async function mount(view) {
  view.innerHTML = `<div class="page">
    <div class="page-head"><div><h2>规则库</h2><p>退改规则原文，检索与生成的唯一依据。修改后索引会自动重建。</p></div>
      <button class="btn btn-primary" type="button" data-add>${icon("plus", "sm")}新建规则</button></div>
    <div class="rules-grid">
      <section class="card" aria-labelledby="rt-h">
        <div class="card-head"><h3 id="rt-h">${icon("book", "sm")}规则 <span class="num muted" data-count></span></h3>
          <div class="row filters">
            <div class="search">${icon("search", "sm")}<input class="input" type="search" data-q placeholder="搜索 ID / 标题 / 内容" aria-label="搜索规则"></div>
            <select class="input" data-f="airline" aria-label="航司">${opts(withAny(AIRLINES), "", "全部航司")}</select>
            <select class="input" data-f="cabin" aria-label="舱位">${opts(withAny(CABINS), "", "全部舱位")}</select>
            <select class="input" data-f="fare_product" aria-label="票价产品">${opts(withAny(PRODUCTS), "", "全部票种")}</select>
            <select class="input" data-f="scenario" aria-label="场景">${opts(SCENARIOS, "", "全部场景")}</select>
          </div></div>
        <div class="table-wrap flat tall" data-table>${skeleton(8, 24)}</div>
      </section>
      <section class="card" aria-labelledby="dbg-h">
        <div class="card-head"><h3 id="dbg-h">${icon("funnel", "sm")}检索调试器</h3><span class="faint">查看漏斗每一层的结果与得分</span></div>
        <form class="card-pad stack" data-debug style="gap:12px">
          <div class="field"><label for="dbg-q">问题</label><textarea id="dbg-q" class="textarea" rows="2" maxlength="1000" required>东方云航折扣经济舱改签手续费怎么算</textarea></div>
          <div class="field-row">
            <div class="field"><label for="dbg-airline">航司过滤</label><select id="dbg-airline" name="airline">${opts(AIRLINES, "", "不过滤")}</select></div>
            <div class="field"><label for="dbg-cabin">舱位过滤</label><select id="dbg-cabin" name="cabin">${opts(CABINS, "", "不过滤")}</select></div>
            <div class="field"><label for="dbg-fp">票种过滤</label><select id="dbg-fp" name="fare_product">${opts(PRODUCTS, "", "不过滤")}</select></div>
            <div class="field"><label for="dbg-sc">场景</label><select id="dbg-sc" name="scenario">${opts(SCENARIOS, "", "不指定")}</select></div>
            <div class="field"><label for="dbg-user">画像加权</label><select id="dbg-user" name="user_id"><option value="">不加权</option>${state.users.map((u) => `<option value="${u.user_id}" ${u.user_id === state.userId ? "selected" : ""}>${esc(u.name)} · ${esc(u.tier)}</option>`).join("")}</select></div>
          </div>
          <div class="row"><button type="submit" class="btn btn-primary">${icon("search", "sm")}检索</button><span class="faint" data-dbg-meta></span></div>
        </form>
        <div class="card-pad" style="padding-top:0" data-funnel>${empty("输入问题后点击检索", "漏斗：向量 60 → RRF 40 → 过滤 → 精排 10 → MMR 5 → 注入 3", "funnel")}</div>
      </section>
    </div></div>`;
  const $ = (s) => view.querySelector(s);
  let rules = [], q = "", hl = null;
  const F = {};
  const draw = () => {
    const rows = rules.filter((r) => !q || [r.id, r.title, r.content].some((x) => x.toLowerCase().includes(q)));
    $("[data-count]").textContent = `${rows.length} / ${rules.length}`;
    if (!rows.length) { $("[data-table]").innerHTML = empty(rules.length ? "没有匹配的规则" : "规则库为空", "调整筛选条件或新建规则", "book"); return; }
    $("[data-table]").innerHTML = `<table class="table"><thead><tr><th>ID</th><th>航司</th><th>舱位</th><th>票种</th><th>场景</th><th>标题 / 内容</th><th><span class="sr-only">操作</span></th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="${r.id === hl ? "selected" : ""}"><td><code>${esc(r.id)}</code></td><td>${esc(r.airline)}</td><td>${esc(r.cabin)}</td><td>${esc(r.fare_product)}</td><td><span class="chip">${esc(SCENARIOS[r.scenario] || r.scenario)}</span></td>
        <td class="clip" style="max-width:420px"><div>${esc(r.title)}</div><div class="faint clip">${esc(r.content)}</div></td>
        <td class="row-actions"><button type="button" class="icon-btn sm" data-edit="${esc(r.id)}" aria-label="编辑 ${esc(r.id)}">${icon("edit", "sm")}</button><button type="button" class="icon-btn sm" data-del="${esc(r.id)}" aria-label="删除 ${esc(r.id)}">${icon("trash", "sm")}</button></td></tr>`).join("")}</tbody></table>`;
    view.querySelectorAll("[data-edit]").forEach((b) => (b.onclick = () => edit(rules.find((r) => r.id === b.dataset.edit))));
    view.querySelectorAll("[data-del]").forEach((b) => (b.onclick = () => remove(b.dataset.del)));
  };
  const load = async () => {
    const qs = new URLSearchParams(Object.entries(F).filter(([, v]) => v)).toString();
    try { rules = (await get(`/api/rules${qs ? `?${qs}` : ""}`)).rules; } catch (e) { rules = []; toastError(e, "规则加载失败"); }
    draw();
  };

  const edit = (r) => {
    const isNew = !r;
    r = r || { id: "", airline: "*", cabin: "*", fare_product: "*", scenario: "VOLUNTARY_CHANGE", title: "", content: "" };
    const form = document.createElement("form");
    form.className = "stack"; form.style.gap = "14px"; form.noValidate = true;
    const fld = (n, label, ctl, help = "") => `<div class="field" data-field="${n}"><label for="rf-${n}">${label}</label>${ctl}${help ? `<span class="help">${help}</span>` : ""}<span class="err" hidden></span></div>`;
    form.innerHTML = `${fld("id", "规则 ID", `<input class="input mono" id="rf-id" name="id" value="${esc(r.id)}" ${isNew ? "" : "readonly"}>`, isNew ? "如 MU-Y-CHG，创建后不可修改" : "ID 不可修改")}
      <div class="field-row">${fld("airline", "航司", `<select id="rf-airline" name="airline">${opts(withAny(AIRLINES), r.airline)}</select>`)}${fld("cabin", "舱位", `<select id="rf-cabin" name="cabin">${opts(withAny(CABINS), r.cabin)}</select>`)}
      ${fld("fare_product", "票价产品", `<select id="rf-fare_product" name="fare_product">${opts(withAny(PRODUCTS), r.fare_product)}</select>`)}${fld("scenario", "场景", `<select id="rf-scenario" name="scenario">${opts(SCENARIOS, r.scenario)}</select>`)}</div>
      ${fld("title", "标题", `<input class="input" id="rf-title" name="title" value="${esc(r.title)}" maxlength="200">`)}
      ${fld("content", "规则原文", `<textarea class="textarea" id="rf-content" name="content" rows="8" maxlength="4000">${esc(r.content)}</textarea>`, "生成的方案只能引用这里出现的说法；金额由运价引擎计算")}`;
    const d = drawer({ title: isNew ? "新建规则" : `编辑 ${r.id}`, labelIcon: "book", width: "620px", body: form,
      foot: `<button type="button" class="btn" data-close>取消</button><button type="button" class="btn btn-primary" data-save>${icon("check", "sm")}保存</button>` });
    d.foot.querySelector("[data-close]").onclick = () => d.close();
    d.foot.querySelector("[data-save]").onclick = async (e) => {
      const body = Object.fromEntries(new FormData(form));
      form.querySelectorAll(".field").forEach((x) => { x.classList.remove("invalid"); x.querySelector(".err").hidden = true; });
      const done = busy(e.currentTarget, "保存中");
      try {
        if (isNew) await post("/api/rules", body); else await put(`/api/rules/${encodeURIComponent(r.id)}`, body);
        d.close(); toast(`已保存规则 ${body.id}，索引将自动重建`, "ok"); hl = body.id; load();
      } catch (err) {
        Object.entries(err.fields || {}).forEach(([k, m]) => { const el = form.querySelector(`[data-field="${k}"]`); if (el) { el.classList.add("invalid"); const s = el.querySelector(".err"); s.hidden = false; s.textContent = m; } });
        toastError(err, "保存失败");
      } finally { done(); }
    };
  };
  const remove = async (id) => {
    if (!(await confirmDialog({ title: "删除规则", message: `确定删除 ${id}？删除后检索将无法命中该规则。`, okText: "删除", danger: true }))) return;
    try { await del(`/api/rules/${encodeURIComponent(id)}`); toast(`已删除 ${id}`, "ok"); load(); } catch (e) { toastError(e); }
  };

  $("[data-add]").onclick = () => edit(null);
  // 点击漏斗里的规则 → 在规则表中高亮
  $("[data-funnel]").addEventListener("click", (ev) => { const li = ev.target.closest("[data-rule]"); if (li) { hl = li.dataset.rule; draw(); view.querySelector("tr.selected")?.scrollIntoView({ block: "nearest" }); } });
  const dq = debounce(draw, 150);
  $("[data-q]").oninput = (e) => { q = e.target.value.trim().toLowerCase(); dq(); };
  view.querySelectorAll("[data-f]").forEach((s) => (s.onchange = () => { F[s.dataset.f] = s.value; load(); }));

  $("[data-debug]").onsubmit = async (e) => {
    e.preventDefault();
    const fd = Object.fromEntries(new FormData(e.target));
    const query = $("#dbg-q").value.trim();
    if (!query) { toast("请输入问题", "warn"); return; }
    const btn = e.target.querySelector("[type=submit]"), done = busy(btn, "检索中");
    $("[data-funnel]").innerHTML = skeleton(7, 28);
    try {
      const filters = Object.fromEntries(["airline", "cabin", "fare_product", "scenario"].filter((k) => fd[k]).map((k) => [k, fd[k]]));
      const r = await post("/api/rules/search-debug", { query, filters, user_id: fd.user_id || null });
      $("[data-dbg-meta]").innerHTML = `<span class="chip ${r.degraded ? "warn" : "ok"}">${icon(r.degraded ? "alert" : "layers", "sm")}${esc(r.backend)}${r.degraded ? " 降级" : ""}</span> <span class="num">${ms(r.ms)}</span>${(r.boost?.airlines || []).length || (r.boost?.products || []).length ? ` · 加权 ${esc([...(r.boost.airlines || []), ...(r.boost.products || [])].join("、"))}` : ""}`;
      renderFunnel($("[data-funnel]"), r.funnel, { emptyHint: "没有任何规则被召回" });
    } catch (err) { $("[data-funnel]").innerHTML = empty("检索失败", err.message, "alert"); toastError(err, "检索失败"); }
    finally { done(); }
  };
  await load();
}
