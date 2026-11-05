// 节点时间线渲染器：对话页（实时追加）与评测下钻（静态）共用
import { icon } from "./icons.js";
import { esc, jsonView, ms, statusTag, empty } from "./ui.js";

export const NODE = {
  user_input: ["接收诉求", "chat"], memory_node: ["加载记忆", "brain"], order_align: ["订单对齐", "ticket"],
  query_understand: ["Query 理解", "sparkles"], clarify: ["追问澄清", "info"], agent_node: ["ReAct 规划", "workflow"],
  schema_validate: ["参数校验", "shield"], checkpoint_gate: ["必经节点检查", "filter"], tool_exec: ["工具执行", "zap"],
  step_back: ["Step-back 改写", "rotate"], sub_done: ["子诉求完成", "check"], tool_scheme_gen: ["方案生成", "layers"],
  verify_node: ["质量校验", "gauge"], retry_node: ["自我修正", "refresh"], human_review: ["转人工", "headset"],
  finish: ["写入记忆", "database"], await_confirm: ["等待确认", "clock"], tool_issue_ticket: ["出票办理", "ticket"],
};
const TOOL = { tool_rule_retrieve: "规则检索", tool_recalc_fee: "费用计算", tool_flight_status: "航班状态", finish: "结束子诉求", clarify: "追问", human: "转人工" };

export const nodeLabel = (n) => (NODE[n] || [n])[0];

/** 下一步阶段提示（流式等待时显示） */
export function stageHint(evt) {
  if (!evt) return "正在理解你的问题…";
  const a = evt.output && evt.output.action;
  switch (evt.node) {
    case "user_input": return "正在加载会话记忆…";
    case "memory_node": return "正在识别订单…";
    case "order_align": return "正在理解问题…";
    case "query_understand": return "正在规划下一步…";
    case "agent_node":
      return a === "tool_rule_retrieve" ? "正在检索规则…" : a === "tool_recalc_fee" ? "正在计算费用…" : a === "tool_flight_status" ? "正在查询航班状态…" : "正在整理结果…";
    case "schema_validate": case "checkpoint_gate": return "正在调用工具…";
    case "tool_exec": case "step_back": return "正在规划下一步…";
    case "sub_done": return "正在生成方案…";
    case "tool_scheme_gen": case "retry_node": return "正在校验金额与依据…";
    case "verify_node": return "正在写入记忆…";
    case "await_confirm": return "正在办理…";
    default: return "处理中…";
  }
}

const STATUS_KIND = { ok: "ok", error: "error", blocked: "warn", human: "human", fallback: "fallback", run: "run" };
const STATUS_TEXT = { ok: "成功", error: "失败", blocked: "被拦截", human: "转人工", fallback: "已降级", run: "进行中" };

function subtitle(e) {
  const tool = e.input && e.input.tool;
  if (e.node === "tool_exec" || e.node === "schema_validate" || e.node === "checkpoint_gate") return tool ? TOOL[tool] || tool : "";
  if (e.node === "agent_node" && e.output && e.output.action) return `→ ${TOOL[e.output.action] || e.output.action}`;
  return "";
}

function item(e, i) {
  const [label, ic] = NODE[e.node] || [e.node, "circle"];
  const st = e.status || "ok";
  const kind = STATUS_KIND[st] || (st === "ok" ? "ok" : "warn");
  const sub = subtitle(e);
  const fallback = /fallback|降级/.test(e.note || "") ? `<span class="chip warn">${icon("alert", "sm")}本步已降级为规则模式</span>` : "";
  return `<li class="tl-item tl-${kind}" data-i="${i}">
    <div class="tl-dot" aria-hidden="true">${icon(ic, "sm")}</div>
    <div class="tl-main">
      <button type="button" class="tl-head" aria-expanded="false">
        <span class="tl-title">${esc(label)}${sub ? ` <span class="faint">${esc(sub)}</span>` : ""}</span>
        <span class="tl-meta">${statusTag(kind, STATUS_TEXT[st] || st)}<span class="num faint">${ms(e.ms)}</span>${icon("chevronDown", "sm tl-chev")}</span>
      </button>
      ${e.note ? `<div class="tl-note">${esc(e.note)}</div>` : ""}${fallback}
      <div class="tl-detail" hidden></div>
    </div></li>`;
}

/** 静态渲染一组 trace 事件 */
export function renderTrace(container, events, { pending = null } = {}) {
  if (!events || !events.length) {
    container.innerHTML = pending ? "" : empty("暂无执行轨迹", "发送一条消息后，这里会实时点亮每个节点", "workflow");
    if (!pending) return;
  }
  const total = (events || []).reduce((s, e) => s + (e.ms || 0), 0);
  container.innerHTML = `<div class="tl-sum row between"><span class="muted">${events.length} 个节点</span><span class="num muted">合计 ${ms(total)}</span></div>
    <ol class="timeline">${events.map(item).join("")}${pending ? `<li class="tl-item tl-run"><div class="tl-dot" aria-hidden="true">${icon("loader", "sm spin")}</div><div class="tl-main"><div class="tl-head static"><span class="tl-title">${esc(pending)}</span>${statusTag("run")}</div></div></li>` : ""}</ol>`;
  container.querySelectorAll(".tl-item[data-i] .tl-head").forEach((btn) => {
    btn.onclick = () => {
      const li = btn.closest(".tl-item");
      const det = li.querySelector(".tl-detail");
      const open = det.hidden;
      if (open && !det.dataset.filled) {
        const e = events[+li.dataset.i];
        det.innerHTML = `<div class="tl-io"><div><div class="tl-io-label">入参</div>${jsonView(e.input)}</div><div><div class="tl-io-label">出参</div>${jsonView(e.output)}</div></div>${e.at ? `<div class="faint num" style="font-size:12px">${esc(e.at)}</div>` : ""}`;
        det.dataset.filled = "1";
      }
      det.hidden = !open;
      btn.setAttribute("aria-expanded", String(open));
      li.classList.toggle("open", open);
    };
  });
  const last = container.querySelector(".tl-item:last-child");
  if (pending && last) last.scrollIntoView({ block: "nearest" });
}
