// 对话流里的卡片：气泡、方案卡、追问卡、转人工卡、出票成功卡、确认模态框
import { icon } from "../icons.js";
import { esc, money, modal, uid, INTENT } from "../ui.js";

const TYPE_ICON = { CHANGE: "calendar", REFUND: "ticket", POLICY: "book", STATUS: "plane" };
export const ACTION_TEXT = { CHANGE: "确认改签", REFUND: "确认退票" };

export function userBubble(text) {
  return `<div class="msg user"><div class="bubble">${esc(text)}</div></div>`;
}

export function botWrap(inner, cls = "") {
  return `<div class="msg bot ${cls}"><div class="avatar" aria-hidden="true">${icon("sparkles", "sm")}</div><div class="msg-body"><div class="msg-label">AI 助手</div>${inner}</div></div>`;
}

export function thinking(text) {
  return botWrap(`<div class="bubble thinking" role="status" aria-live="polite">${icon("loader", "sm spin")}<span data-hint>${esc(text)}</span></div>`, "pending");
}

function feeTable(o) {
  if (!o.fee_items || !o.fee_items.length) return "";
  return `<table class="fee-table"><tbody>${o.fee_items.map((f) => `<tr><td>${esc(f.name)}</td><td class="num">${money(f.amount)}</td></tr>`).join("")}</tbody></table>`;
}

export function ruleChips(refs) {
  if (!refs || !refs.length) return "";
  return `<div class="rule-chips"><span class="faint">依据</span>${refs.map((r) => `<button type="button" class="chip info" data-rule="${esc(r)}" title="在检索面板中查看">${icon("book", "sm")}${esc(r)}</button>`).join("")}</div>`;
}

/** 方案卡；index 为 final.options 中的下标 */
export function schemeCard(o, index) {
  const actionable = o.allowed !== false && (o.type === "CHANGE" || o.type === "REFUND");
  const totalLabel = o.type === "REFUND" ? "预计退款" : "合计需支付";
  return `<article class="scheme ${o.allowed === false ? "denied" : ""}" data-index="${index}">
    <header class="row between">
      <span class="row"><span class="scheme-ic">${icon(TYPE_ICON[o.type] || "info", "sm")}</span><span class="chip">${esc(INTENT[o.type] || o.type)}</span><h4>${esc(o.title)}</h4></span>
      ${o.allowed === false ? `<span class="status bad">${icon("xCircle", "sm")}不可办理</span>` : actionable ? `<span class="status ok">${icon("checkCircle", "sm")}可办理</span>` : ""}
    </header>
    ${feeTable(o)}
    ${o.total !== null && o.total !== undefined && actionable ? `<div class="scheme-total"><span class="muted">${totalLabel}</span><span class="big num">${money(o.total)}</span></div>` : ""}
    ${o.notes ? `<p class="scheme-notes">${esc(o.notes)}</p>` : ""}
    ${ruleChips(o.rule_refs)}
    ${actionable ? `<div class="scheme-actions"><button type="button" class="btn btn-primary" data-confirm="${index}" disabled aria-disabled="true">${icon("check", "sm")}${ACTION_TEXT[o.type]}</button><span class="faint scheme-hint" data-hint-for="${index}">等待方案校验通过…</span></div>` : ""}
  </article>`;
}

export function clarifyCard(p) {
  return `<div class="clarify card-in" data-clarify>
    <div class="row">${icon("info", "sm")}<strong>${esc(p.question || "需要补充信息")}</strong></div>
    <div class="clarify-opts" role="group" aria-label="选择一个答案">${(p.options || []).map((o, i) => `<button type="button" class="btn btn-sm" data-opt="${i}">${esc(o.label)}</button>`).join("")}</div>
    <p class="faint">也可以直接在输入框里回答</p>
  </div>`;
}

export function humanCard(final, verify) {
  const reasons = [...(final.reasons || []), ...((verify && verify.errors) || [])].filter((x, i, a) => x && a.indexOf(x) === i);
  return `<div class="handoff card-in">
    <div class="row">${icon("headset")}<strong>已转人工客服</strong><span class="chip warn">${icon("alert", "sm")}自动答复未通过检查</span></div>
    <p>${esc(final.text || "")}</p>
    ${reasons.length ? `<div class="faint">未通过的检查</div><ul class="issue-list">${reasons.map((r) => `<li class="bad">${icon("xCircle", "sm")}<span>${esc(r)}</span></li>`).join("")}</ul>` : ""}
  </div>`;
}

export function ticketCard(final) {
  const t = final.ticket || {};
  return `<div class="ticket-ok card-in">
    <div class="ticket-ic" aria-hidden="true">${icon("checkCircle", "lg")}</div>
    <div class="stack" style="gap:4px">
      <strong>${t.action === "REFUND" ? "退票已受理" : "改签成功"}</strong>
      <p>${esc(final.text || "")}</p>
      <dl class="kv">
        ${t.order_id ? `<dt>订单</dt><dd><code>${esc(t.order_id)}</code></dd>` : ""}
        ${t.ticket_no ? `<dt>新票号</dt><dd class="num">${esc(t.ticket_no)}</dd>` : ""}
        ${t.target ? `<dt>新航班</dt><dd>${esc(t.target.flight_no)} · ${esc(t.target.date)}</dd>` : ""}
        ${t.amount !== undefined ? `<dt>${t.action === "REFUND" ? "退款" : "支付"}</dt><dd class="num big-sm">${money(t.amount)}</dd>` : ""}
        ${t.confirm_id ? `<dt>确认号</dt><dd class="num faint">${esc(t.confirm_id)}</dd>` : ""}
      </dl>
    </div></div>`;
}

export function errorCard(err) {
  const cfg = err.code === "LLM_NOT_CONFIGURED";
  return `<div class="callout bad card-in" role="alert">${icon("xCircle")}<div><strong>${cfg ? "DeepSeek 未配置" : "处理失败"}</strong><p>${esc(err.message || "未知错误")}</p>
    ${cfg ? `<p style="margin-top:6px"><a href="#/config">前往配置中心填写 API Key</a>，或在顶栏切回「规则模式」。</p>` : ""}</div></div>`;
}

/** 出票确认模态框：金额明细 + 不可撤销警告。onConfirm(confirm_id) 返回 Promise */
export function confirmModal(option, order, onConfirm) {
  const verb = option.type === "REFUND" ? "退票" : "改签";
  const confirmId = uid("c-");
  const m = modal({
    title: `确认${verb}`, labelIcon: option.type === "REFUND" ? "ticket" : "calendar",
    body: `<p class="muted">${esc(order ? `${order.flight_no} ${order.flight?.origin_city || ""}→${order.flight?.dest_city || ""} ${order.date} · ${order.product_label}` : option.title)}</p>
      <div class="confirm-box"><h4>${esc(option.title)}</h4>
        <table class="fee-table"><tbody>${(option.fee_items || []).map((f) => `<tr><td>${esc(f.name)}</td><td class="num">${money(f.amount)}</td></tr>`).join("")}</tbody>
        <tfoot><tr><td>${option.type === "REFUND" ? "预计退款" : "合计需支付"}</td><td class="num big">${money(option.total)}</td></tr></tfoot></table></div>
      <div class="callout warn">${icon("alert")}<div><strong>操作不可撤销</strong><p>确认后将立即${verb}，原${option.type === "REFUND" ? "客票作废" : "航班座位释放"}。确认号 <code>${esc(confirmId)}</code> 保证重复提交只办理一次。</p></div></div>`,
    foot: `<button class="btn" type="button" data-close>再想想</button><button class="btn btn-primary" type="button" data-ok autofocus>${icon("check", "sm")}确认${verb} ${money(option.total)}</button>`,
  });
  m.foot.querySelector("[data-close]").onclick = () => m.close();
  m.foot.querySelector("[data-ok]").onclick = () => { m.close(); onConfirm(confirmId); };
  return m;
}
