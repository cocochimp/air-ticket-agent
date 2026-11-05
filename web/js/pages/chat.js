// 对话工作台：订单卡片条 + 对话流 + 快捷示例 + 输入框；右侧执行面板（流程/检索/记忆/校验）
import { get, stream } from "../api.js";
import { icon } from "../icons.js";
import { state, on, setUser } from "../store.js";
import { stageHint } from "../trace.js";
import { esc, toast, toastError, uid } from "../ui.js";
import { createPanel } from "./chat-panel.js";
import { userBubble, botWrap, thinking, schemeCard, clarifyCard, humanCard, ticketCard, errorCard, confirmModal } from "./chat-cards.js";

const EXAMPLES = [
  { label: "改签差价", user: "u_zhang", order: (os) => os.find((o) => o.order_id.endsWith("001")), text: "帮我改到大后天" },
  { label: "改签+退票", user: "u_zhang", order: (os) => os.find((o) => o.order_id.endsWith("001")), text: "改到大后天，顺便看看退票多少钱" },
  { label: "航班取消", user: "u_wang", order: (os) => os.find((o) => o.flight_status === "cancelled"), text: "航班取消了，帮我退票" },
  { label: "指代追问", user: "u_zhang", order: () => null, text: "回程那张能退吗" },
  { label: "模糊日期", user: "u_zhang", order: (os) => os.find((o) => o.order_id.endsWith("001")), text: "改到下个月初" },
  { label: "转人工", user: "u_zhang", order: (os) => os.find((o) => o.order_id.endsWith("001")), text: "宠物能带上飞机吗" },
];
const FLIGHT_CHIP = { scheduled: ["ok", "checkCircle", "正常"], delayed: ["warn", "clock", "延误"], cancelled: ["bad", "xCircle", "取消"] };
const ORDER_CHIP = { ISSUED: "", CHANGED: "info", REFUNDED: "bad" };

const sessionKey = (u) => `session:${u}`;
function sessionFor(u, fresh = false) {
  let s = !fresh && localStorage.getItem(sessionKey(u));
  if (!s) { s = uid(`s-${u}-`); localStorage.setItem(sessionKey(u), s); }
  return s;
}

export async function mount(view) {
  view.innerHTML = `<div class="chat-page">
    <section class="chat-col" aria-label="对话">
      <div class="orders-strip" role="group" aria-label="选择订单"></div>
      <div class="chat-stream" aria-live="polite" aria-label="对话记录"></div>
      <div class="composer">
        <div class="examples" role="group" aria-label="快捷示例">${EXAMPLES.map((e, i) => `<button type="button" class="chip-btn" data-ex="${i}" title="${esc(e.text)}">${icon("zap", "sm")}${esc(e.label)}</button>`).join("")}</div>
        <form class="input-row" autocomplete="off">
          <label for="chat-input" class="sr-only">输入你的问题</label>
          <textarea id="chat-input" rows="1" maxlength="1000" placeholder="描述你的退改需求，Enter 发送，Shift+Enter 换行"></textarea>
          <div class="input-actions">
            <span class="num faint counter" aria-live="off">0/1000</span>
            <button type="button" class="icon-btn" data-new aria-label="新建会话" title="新建会话">${icon("plus")}</button>
            <button type="submit" class="btn btn-primary send-btn" aria-label="发送">${icon("send", "sm")}发送</button>
          </div>
        </form>
      </div>
    </section>
    <aside class="exec-panel" aria-label="执行面板"><div class="exec-inner"></div></aside>
    <button type="button" class="btn panel-toggle" aria-expanded="false" aria-controls="exec-panel">${icon("panel", "sm")}执行面板</button>
  </div>`;
  const $ = (s) => view.querySelector(s);
  const streamEl = $(".chat-stream"), strip = $(".orders-strip"), input = $("#chat-input"), form = $(".input-row");
  const panelEl = $(".exec-panel"), toggle = $(".panel-toggle");
  panelEl.id = "exec-panel";
  let cfg = {};
  try { cfg = await get("/api/config"); } catch { /* 用默认阈值 */ }
  const panel = createPanel($(".exec-inner"), { threshold: cfg.verify?.threshold ?? 0.85, maxRetries: cfg.verify?.max_retries ?? 1 });

  let orders = [], selected = null, sid = sessionFor(state.userId), busyFlag = false, pending = null, ctrl = null;

  // ---------- 订单卡片 ----------
  const drawOrders = () => {
    if (!orders.length) { strip.innerHTML = `<div class="faint" style="padding:8px">该用户没有订单</div>`; return; }
    strip.innerHTML = orders.map((o) => {
      const [cls, ic, lab] = FLIGHT_CHIP[o.flight_status] || ["", "circle", o.flight_status_label];
      const f = o.flight || {};
      return `<button type="button" class="order-card ${o.order_id === selected ? "selected" : ""}" data-order="${esc(o.order_id)}" aria-pressed="${o.order_id === selected}">
        <span class="oc-top"><span class="oc-flight num">${esc(o.flight_no)}</span><span class="chip ${cls}">${icon(ic, "sm")}${esc(lab)}${o.flight_status === "delayed" ? ` ${o.delay_min}分` : ""}</span></span>
        <span class="oc-route">${esc(f.origin_city || "")}<span class="faint">${icon("arrowRight", "sm")}</span>${esc(f.dest_city || "")}</span>
        <span class="oc-meta num">${esc(o.date.slice(5))} ${esc(f.dep_time || "")}</span>
        <span class="oc-foot"><span class="faint">${esc(o.airline_name)} · ${esc(o.product_label)}</span>${o.status !== "ISSUED" ? `<span class="chip ${ORDER_CHIP[o.status] || ""}">${esc(o.status_label)}</span>` : ""}</span>
      </button>`;
    }).join("");
    strip.querySelectorAll("[data-order]").forEach((b) => (b.onclick = () => { selected = selected === b.dataset.order ? null : b.dataset.order; drawOrders(); }));
  };
  const loadOrders = async () => {
    strip.innerHTML = Array.from({ length: 3 }, () => `<div class="order-card skeleton" style="height:104px"></div>`).join("");
    try { orders = (await get(`/api/orders?user_id=${encodeURIComponent(state.userId)}`)).orders; }
    catch (e) { orders = []; toastError(e, "订单加载失败"); }
    if (!orders.some((o) => o.order_id === selected)) {
      const today = (state.status?.now || new Date().toISOString()).slice(0, 10);
      selected = (orders.find((o) => o.status === "ISSUED" && o.date >= today) || orders[0])?.order_id || null;
    }
    drawOrders();
    return orders;
  };
  // ---------- 对话流 ----------
  const add = (html) => {
    streamEl.querySelector(".chat-empty")?.remove();
    streamEl.insertAdjacentHTML("beforeend", html);
    const el = streamEl.lastElementChild;
    el.scrollIntoView({ block: "end", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    return el;
  };
  const welcome = () => {
    const u = state.users.find((x) => x.user_id === state.userId);
    streamEl.innerHTML = `<div class="chat-empty">${icon("sparkles", "lg")}<h3>你好${u ? `，${esc(u.name)}` : ""}，我是退改签 AI 助手</h3>
      <p class="muted">先在上方选择订单，再描述你的需求；或点击下方快捷示例。所有金额都来自运价引擎，经过校验后才会给出。</p></div>`;
  };
  const lockInput = (on) => {
    busyFlag = on;
    $(".send-btn").disabled = on;
    view.querySelectorAll("[data-ex]").forEach((b) => (b.disabled = on));
  };
  const disableOld = () => {
    streamEl.querySelectorAll("[data-confirm]").forEach((b) => { b.disabled = true; b.setAttribute("aria-disabled", "true"); });
    streamEl.querySelectorAll(".scheme-hint").forEach((h) => (h.textContent = "已失效（有新的对话）"));
    streamEl.querySelectorAll("[data-clarify] button").forEach((b) => (b.disabled = true));
  };

  const bindRuleChips = (el) => el.querySelectorAll("[data-rule]").forEach((b) => (b.onclick = () => { panel.showRule(b.dataset.rule); openPanel(true); }));

  /** 根据 interrupt 渲染可交互部分 */
  const applyInterrupt = (intr, card) => {
    pending = intr || null;
    if (!intr) return;
    if (intr.kind === "confirm" && card) {
      const ok = new Set((intr.options || []).map((o) => o.index));
      card.querySelectorAll("[data-confirm]").forEach((b) => {
        const i = +b.dataset.confirm;
        const hint = card.querySelector(`[data-hint-for="${i}"]`);
        if (ok.has(i)) { b.disabled = false; b.removeAttribute("aria-disabled"); if (hint) hint.textContent = "校验已通过，可办理"; }
        else if (hint) hint.textContent = "该方案不可办理";
        b.onclick = () => openConfirm(intr, i);
      });
      if (!card.querySelector("[data-cancel]")) {
        card.insertAdjacentHTML("beforeend", `<div class="row" style="margin-top:4px"><button type="button" class="btn btn-ghost btn-sm" data-cancel>${icon("x", "sm")}暂不办理</button></div>`);
        card.querySelector("[data-cancel]").onclick = (e) => { e.target.closest("button").disabled = true; resume({ confirm: false }, "暂不办理"); };
      }
    } else if (intr.kind === "clarify") {
      const el = add(botWrap(clarifyCard(intr)));
      el.querySelectorAll("[data-opt]").forEach((b) => (b.onclick = () => {
        const o = intr.options[+b.dataset.opt];
        resume({ value: o.value }, o.label);
      }));
    }
  };

  const renderResult = (res) => {
    const f = res.final || {};
    const intr = res.interrupt;
    if (intr && intr.kind === "clarify") { applyInterrupt(intr); return; }
    let el;
    if (res.outcome === "HUMAN" || f.outcome === "HUMAN") el = add(botWrap(humanCard(f, res.verify)));
    else if (res.outcome === "TICKETED") { el = add(botWrap(ticketCard(f))); loadOrders(); }
    else if (res.outcome === "CANCELLED") el = add(botWrap(`<div class="bubble">${icon("info", "sm")} 已取消办理，订单保持不变。还有其他问题可以继续问我。</div>`));
    else {
      const opts = f.options || [];
      el = add(botWrap(`${f.text ? `<div class="bubble">${esc(f.text)}</div>` : ""}${opts.map((o, i) => schemeCard(o, i)).join("")}`));
      if (!intr) el.querySelectorAll(".scheme-hint").forEach((h) => (h.textContent = "本轮无可办理的确认"));
    }
    if (el) { bindRuleChips(el); applyInterrupt(intr, el); }
  };

  const handlers = (pendingEl) => {
    return {
      trace: (e) => { const h = stageHint(e); panel.push(e, h); const t = pendingEl.querySelector("[data-hint]"); if (t) t.textContent = h; },
      interrupt: () => {},
      final: (res) => { pendingEl.remove(); panel.done(res); renderResult(res); },
      error: (err) => {
        pendingEl.remove(); panel.done(null); add(botWrap(errorCard(err)));
        if (err.code === "NETWORK") restore();
      },
    };
  };

  const run = async (url, body) => {
    lockInput(true);
    disableOld();
    pending = null;
    const pendingEl = add(thinking("正在理解你的问题…"));
    panel.start("正在理解你的问题…");
    ctrl = new AbortController();
    try { await stream("POST", url, body, handlers(pendingEl), ctrl.signal); }
    finally { if (pendingEl.isConnected) { pendingEl.remove(); panel.done(null); } lockInput(false); input.focus(); }
  };

  const send = (text) => {
    text = text.trim();
    if (!text || busyFlag) return;
    if (text.length > 1000) { toast("消息不能超过 1000 字", "warn"); return; }
    add(userBubble(text));
    if (pending && pending.kind === "clarify") return run("/api/resume", { session_id: sid, payload: { value: text } });
    return run("/api/chat", { session_id: sid, user_id: state.userId, order_id: selected, message: text, mode: state.mode });
  };
  const resume = (payload, label) => {
    if (busyFlag) return;
    if (label) add(userBubble(label));
    return run("/api/resume", { session_id: sid, payload });
  };
  const openConfirm = (intr, i) => {
    const opt = (intr.options || []).find((o) => o.index === i);
    if (!opt) return;
    const order = orders.find((o) => o.order_id === intr.order_id);
    confirmModal(opt, order, (confirmId) => resume({ confirm: true, index: i, confirm_id: confirmId }, `确认${opt.type === "REFUND" ? "退票" : "改签"}`));
  };

  /** 恢复会话：拉取 pending interrupt 与记忆（页面刷新或 SSE 断开后） */
  const restore = async () => {
    try {
      const s = await get(`/api/sessions/${encodeURIComponent(sid)}`);
      panel.setMemory(s.memory);
      if (s.pending_interrupt) {
        const intr = s.pending_interrupt;
        if (intr.kind === "confirm") {
          const opts = (intr.options || []);
          const el = add(botWrap(`<div class="bubble">${icon("history", "sm")} 已恢复上次未完成的办理：${esc(intr.message || "")}</div>${opts.map((o) => schemeCard(o, o.index)).join("")}`));
          bindRuleChips(el);
          applyInterrupt(intr, el);
        } else applyInterrupt(intr);
      }
    } catch { /* 新会话 */ }
  };

  // ---------- 输入 ----------
  const counter = $(".counter");
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(160, input.scrollHeight) + "px"; counter.textContent = `${input.value.length}/1000`; };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  form.onsubmit = (e) => { e.preventDefault(); const v = input.value; input.value = ""; autosize(); send(v); };
  $("[data-new]").onclick = () => {
    if (busyFlag) return;
    sid = sessionFor(state.userId, true);
    pending = null; panel.reset(); welcome(); toast("已开始新会话", "ok");
  };
  view.querySelectorAll("[data-ex]").forEach((b) => (b.onclick = async () => {
    if (busyFlag) return;
    const ex = EXAMPLES[+b.dataset.ex];
    if (state.userId !== ex.user) { setUser(ex.user); await switching; }
    const o = ex.order(orders);
    selected = o ? o.order_id : null;
    drawOrders();
    send(ex.text);
  }));

  // ---------- 面板抽屉（<1024px） ----------
  const openPanel = (on) => {
    if (!matchMedia("(max-width: 1023px)").matches) return;
    panelEl.classList.toggle("open", on);
    toggle.setAttribute("aria-expanded", String(on));
  };
  toggle.onclick = () => openPanel(!panelEl.classList.contains("open"));
  const onKey = (e) => { if (e.key === "Escape" && panelEl.classList.contains("open")) openPanel(false); };
  document.addEventListener("keydown", onKey);

  // ---------- 用户切换 ----------
  let switching = Promise.resolve();
  const off = on((key) => {
    if (key === "user") {
      ctrl?.abort();
      sid = sessionFor(state.userId);
      pending = null; selected = null; panel.reset(); welcome();
      switching = loadOrders().then(restore);
    }
    if (key === "users" && !streamEl.querySelector(".msg")) welcome();
  });

  welcome();
  await loadOrders();
  await restore();
  input.focus();
  return () => { off(); ctrl?.abort(); document.removeEventListener("keydown", onKey); };
}
