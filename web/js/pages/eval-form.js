// 评测用例编辑抽屉：分组表单 + 字段级错误（来自 /api/eval/cases/validate）
import { post } from "../api.js";
import { icon } from "../icons.js";
import { esc, drawer, busy, toast, toastError, AIRLINES, PRODUCTS, INTENT, OUTCOME } from "../ui.js";

const FORBID = { quote_mismatch: "金额与引擎不符", unsupported_claim: "无依据论断", skip_rule: "跳过检索直接算费" };
const TOOLS = ["tool_rule_retrieve", "tool_recalc_fee", "tool_flight_status"];
const split = (s) => s.split(/[,，\s]+/).map((x) => x.trim()).filter(Boolean);
const optList = (map, cur, blank = "") => `${blank ? `<option value="">${blank}</option>` : ""}${Object.entries(map).map(([k, v]) => `<option value="${esc(k)}" ${k === cur ? "selected" : ""}>${esc(k)} · ${esc(v)}</option>`).join("")}`;

function f(path, label, control, help = "") {
  return `<div class="field" data-field="${path}"><label for="cf-${path}">${label}</label>${control}${help ? `<span class="help">${esc(help)}</span>` : ""}<span class="err" hidden></span></div>`;
}
const text = (path, v, attrs = "") => `<input class="input" id="cf-${path}" name="${path}" value="${esc(v ?? "")}" ${attrs}>`;
const area = (path, v, rows = 3, mono = false) => `<textarea class="textarea ${mono ? "mono" : ""}" id="cf-${path}" name="${path}" rows="${rows}">${esc(v ?? "")}</textarea>`;
const checks = (path, map, cur) => `<div class="row" id="cf-${path}" role="group">${Object.entries(map).map(([k, v]) => `<label class="checkbox"><input type="checkbox" name="${path}" value="${k}" ${cur.includes(k) ? "checked" : ""}>${esc(v)}</label>`).join("")}</div>`;

function formHTML(c, users) {
  const e = c.expect || {};
  const fees = e.fees || {};
  return `<div class="callout bad" data-summary hidden>${icon("xCircle")}<div><strong>校验未通过</strong><ul class="issue-list" style="margin-top:4px"></ul></div></div>
  <fieldset><legend>${icon("info", "sm")}基本信息</legend><div class="field-row">
    ${f("id", "用例 ID", text("id", c.id, 'required pattern="[\\w-]{1,64}"'), "字母数字下划线，唯一")}
    ${f("title", "标题", text("title", c.title))}
    ${f("tags", "标签", text("tags", (c.tags || []).join(", ")), "逗号分隔，如 change, multi")}
  </div></fieldset>
  <fieldset><legend>${icon("user", "sm")}用户与订单</legend><div class="field-row">
    ${f("user_id", "用户", `<select id="cf-user_id" name="user_id">${users.map((u) => `<option value="${esc(u.user_id)}" ${u.user_id === c.user_id ? "selected" : ""}>${esc(u.name)} · ${esc(u.tier)}</option>`).join("")}</select>`)}
    ${f("airline", "航司", `<select id="cf-airline" name="airline">${optList(AIRLINES, c.airline, "（不限）")}</select>`)}
    ${f("fare_product", "票价产品", `<select id="cf-fare_product" name="fare_product">${optList(PRODUCTS, c.fare_product, "（不限）")}</select>`)}
    ${f("order", "订单", text("order", c.order, 'placeholder="#1"'), "#n 表示第 n 个演示订单；留空表示不预选")}
  </div>
  ${f("question", "乘客问题", area("question", c.question, 2))}
  ${f("history", "历史轮次", area("history", JSON.stringify(c.history || [], null, 2), 3, true), '[{"message":"…","order":"#2"}]，在本问题之前依次发送')}
  ${f("clarify_answers", "追问答案", text("clarify_answers", (Array.isArray(c.clarify_answers) ? c.clarify_answers : []).join(", ")), "按出现顺序回答追问，逗号分隔，如 2026-10-02 或 #3")}
  </fieldset>
  <fieldset><legend>${icon("target", "sm")}期望结果</legend>
    ${f("expect.intents", "意图", checks("expect.intents", INTENT, e.intents || []))}
    ${f("expect.tools", "工具序列（有序）", text("expect.tools", (e.tools || []).join(", "), `list="tool-names"`), TOOLS.join(" / "))}
    <datalist id="tool-names">${TOOLS.map((t) => `<option value="${t}">`).join("")}</datalist>
    <div class="field-row">
      ${f("expect.fees.CHANGE", "改签应付（元）", text("expect.fees.CHANGE", fees.CHANGE ?? "", 'inputmode="numeric" class="input num"'), "留空表示不校验")}
      ${f("expect.fees.REFUND", "退票应退（元）", text("expect.fees.REFUND", fees.REFUND ?? "", 'inputmode="numeric"'), "由运价引擎重算比对")}
      ${f("expect.outcome", "结局", `<select id="cf-expect.outcome" name="expect.outcome">${optList({ ANSWER: OUTCOME.ANSWER, HUMAN: OUTCOME.HUMAN, CLARIFY: OUTCOME.CLARIFY }, e.outcome || "ANSWER")}</select>`)}
    </div>
    ${f("expect.rule_ids", "必须命中的规则 ID", text("expect.rule_ids", (e.rule_ids || []).join(", ")), "必须全部出现在前 3 条检索结果中")}
    ${f("expect.schema_errors", "期望的 Schema 错误", area("expect.schema_errors", e.schema_errors == null ? "" : JSON.stringify(e.schema_errors), 1, true), "可选，JSON")}
  </fieldset>
  <fieldset><legend>${icon("shield", "sm")}禁止出现</legend>${f("forbid", "以下任一出现即判失败", checks("forbid", FORBID, c.forbid || []))}</fieldset>`;
}
/** 读表单 → 用例对象；JSON 解析失败的字段放进 errs */
function readForm(root, base) {
  const val = (n) => root.querySelector(`[name="${n}"]`)?.value.trim() ?? "";
  const checked = (n) => [...root.querySelectorAll(`[name="${n}"]:checked`)].map((x) => x.value);
  const errs = [];
  const json = (n, dflt) => { const s = val(n); if (!s) return dflt; try { return JSON.parse(s); } catch { errs.push({ field: n, message: "不是合法的 JSON" }); return dflt; } };
  const num = (n) => { const s = val(n); if (s === "") return null; const v = Number(s); if (!Number.isInteger(v)) errs.push({ field: n, message: "必须是整数" }); return v; };
  const fees = {};
  for (const k of ["CHANGE", "REFUND"]) { const v = num(`expect.fees.${k}`); if (v !== null) fees[k] = v; }
  const c = {
    ...base, id: val("id"), title: val("title"), tags: split(val("tags")), user_id: val("user_id"),
    airline: val("airline") || null, fare_product: val("fare_product") || null, order: val("order") || null,
    question: val("question"), history: json("history", []), clarify_answers: split(val("clarify_answers")),
    expect: { ...(base.expect || {}), intents: checked("expect.intents"), tools: split(val("expect.tools")), fees,
      rule_ids: split(val("expect.rule_ids")), outcome: val("expect.outcome") || "ANSWER", schema_errors: json("expect.schema_errors", null) },
    forbid: checked("forbid"),
  };
  if (c.expect.schema_errors === null) delete c.expect.schema_errors;
  return { c, errs };
}

function showErrors(root, errs) {
  root.querySelectorAll(".field.invalid").forEach((x) => { x.classList.remove("invalid"); const e = x.querySelector(".err"); e.hidden = true; });
  const summary = root.querySelector("[data-summary]");
  const loose = [];
  for (const e of errs) {
    let el = root.querySelector(`[data-field="${CSS.escape(e.field)}"]`);
    if (!el && e.field.startsWith("expect.fees")) el = root.querySelector(`[data-field="${CSS.escape(e.field.split(".").slice(0, 3).join("."))}"]`);
    if (!el) el = root.querySelector(`[data-field="${CSS.escape(e.field.split(".").slice(0, 2).join("."))}"]`) || root.querySelector(`[data-field="${CSS.escape(e.field.split(".")[0])}"]`);
    if (el) { el.classList.add("invalid"); const s = el.querySelector(".err"); s.hidden = false; s.innerHTML = `${icon("alert", "sm")}${esc(e.message)}`; }
    else loose.push(e);
  }
  summary.hidden = !errs.length;
  summary.querySelector("ul").innerHTML = errs.map((e) => `<li class="bad">${icon("xCircle", "sm")}<span><code>${esc(e.field)}</code> ${esc(e.message)}</span></li>`).join("");
  root.querySelector(".invalid input, .invalid textarea, .invalid select")?.focus();
  void loose;
}

/** 打开编辑抽屉。onSave(case) 返回 Promise；isNew 时校验 ID 唯一 */
export function openCaseEditor({ c, users, existingIds, isNew, onSave }) {
  const form = document.createElement("form");
  form.className = "stack case-form";
  form.style.gap = "16px";
  form.noValidate = true;
  form.innerHTML = formHTML(c, users);
  const d = drawer({
    title: isNew ? "新建用例" : `编辑用例 ${c.id}`, labelIcon: "edit", width: "720px", body: form,
    foot: `<button type="button" class="btn" data-validate>${icon("shield", "sm")}校验</button><span class="spacer"></span><button type="button" class="btn" data-close>取消</button><button type="button" class="btn btn-primary" data-save>${icon("check", "sm")}保存</button>`,
  });
  const validate = async () => {
    const { c: nc, errs } = readForm(form, c);
    if (!nc.id) errs.push({ field: "id", message: "必填" });
    if (!nc.question) errs.push({ field: "question", message: "必填" });
    if (nc.id !== c.id && existingIds.has(nc.id)) errs.push({ field: "id", message: "用例 ID 已存在" });
    if (errs.length) { showErrors(form, errs); return null; }
    const r = await post("/api/eval/cases/validate", { cases: [nc] });
    const list = Object.values(r.errors || {}).flat();
    showErrors(form, list);
    return list.length ? null : nc;
  };
  d.foot.querySelector("[data-close]").onclick = () => d.close();
  d.foot.querySelector("[data-validate]").onclick = async (e) => {
    const done = busy(e.currentTarget, "校验中");
    try { if (await validate()) toast("校验通过：订单、规则、工具名与引擎金额均一致", "ok"); } catch (err) { toastError(err, "校验失败"); } finally { done(); }
  };
  d.foot.querySelector("[data-save]").onclick = async (e) => {
    const done = busy(e.currentTarget, "保存中");
    try { const nc = await validate(); if (nc) { await onSave(nc, c.id); d.close(); } } catch (err) { toastError(err, "保存失败"); } finally { done(); }
  };
  return d;
}
