// 共享 UI 组件：转义、格式化、Toast、模态框（焦点陷阱）、抽屉、JSON 视图、空状态
import { icon } from "./icons.js";

export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const money = (v) => (v === null || v === undefined ? "—" : `¥${Number(v).toLocaleString("zh-CN")}`);
export const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);
export const ms = (v) => (v === null || v === undefined ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${Math.round(v)}ms`);
export const uid = (p = "") => p + (crypto.randomUUID ? crypto.randomUUID().replace(/-/g, "").slice(0, 16) : Math.random().toString(36).slice(2, 18));

export const INTENT = { CHANGE: "改签", REFUND: "退票", POLICY: "政策咨询", STATUS: "航班状态" };
export const OUTCOME = { ANSWER: "已答复", HUMAN: "转人工", TICKETED: "已办理", CANCELLED: "已取消", CLARIFY: "追问" };
export const AIRLINES = { MU: "东方云航", CA: "国际星航", CZ: "南方海航", HU: "海岛航空" };
export const PRODUCTS = { F: "头等舱全价", C: "公务舱全价", Y: "经济舱全价", Y_DISCOUNT: "经济舱折扣", SPECIAL: "经济舱特价", GROUP: "团队票" };
export const SCENARIOS = { VOLUNTARY_CHANGE: "自愿改签", VOLUNTARY_REFUND: "自愿退票", CANCEL: "航班取消", DELAY: "航班延误", NO_SHOW: "误机", FARE_DIFF: "改签差价" };

/** 状态 = 图标 + 文字，不只靠颜色 */
const STATUS = {
  ok: ["checkCircle", "成功"], error: ["xCircle", "失败"], bad: ["xCircle", "失败"], human: ["headset", "转人工"],
  fallback: ["alert", "已降级"], warn: ["alert", "注意"], run: ["loader", "进行中"], idle: ["circle", "等待"],
};
export function statusTag(kind, text) {
  const [ic, label] = STATUS[kind] || STATUS.idle;
  const cls = { error: "bad", human: "warn", fallback: "warn" }[kind] || kind;
  return `<span class="status ${cls}">${icon(ic, `sm${kind === "run" ? " spin" : ""}`)}${esc(text ?? label)}</span>`;
}
export const outcomeChip = (o) => {
  const cls = { ANSWER: "info", TICKETED: "ok", HUMAN: "warn", CANCELLED: "", CLARIFY: "info" }[o] ?? "";
  const ic = { ANSWER: "chat", TICKETED: "checkCircle", HUMAN: "headset", CANCELLED: "x", CLARIFY: "info" }[o] || "circle";
  return `<span class="chip ${cls}">${icon(ic, "sm")}${esc(OUTCOME[o] || o || "—")}</span>`;
};
export const passChip = (ok) => (ok ? `<span class="chip ok">${icon("check", "sm")}通过</span>` : `<span class="chip bad">${icon("x", "sm")}失败</span>`);

export function empty(title, hint = "", ic = "info") {
  return `<div class="empty">${icon(ic)}<strong>${esc(title)}</strong>${hint ? `<span>${esc(hint)}</span>` : ""}</div>`;
}
export const skeleton = (rows = 3, h = 16) => `<div class="stack">${Array.from({ length: rows }, (_, i) => `<div class="skeleton" style="height:${h}px;width:${90 - i * 12}%"></div>`).join("")}</div>`;

export function jsonView(v) {
  const s = esc(JSON.stringify(v ?? null, null, 2));
  const html = s.replace(/(&quot;(?:[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?)/g, (m, str, colon, lit, num) => {
    if (str) return colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`;
    if (lit) return `<span class="b">${lit}</span>`;
    return `<span class="n">${num}</span>`;
  });
  return `<pre class="json">${html}</pre>`;
}

// ---------- Toast ----------
export function toast(message, kind = "info", title = "") {
  const root = document.getElementById("toasts");
  const ic = { ok: "checkCircle", bad: "xCircle", warn: "alert", info: "info" }[kind] || "info";
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.setAttribute("role", kind === "bad" ? "alert" : "status");
  el.innerHTML = `${icon(ic)}<div class="t-body">${title ? `<div class="t-title">${esc(title)}</div>` : ""}<div>${esc(message)}</div></div>
    <button class="icon-btn sm btn-ghost" type="button" aria-label="关闭提示">${icon("x", "sm")}</button>`;
  const close = () => { el.classList.add("leaving"); setTimeout(() => el.remove(), 160); };
  el.querySelector("button").onclick = close;
  root.appendChild(el);
  setTimeout(close, kind === "bad" ? 7000 : 3500);
}
export const toastError = (e, title = "操作失败") => toast(e?.message || String(e), "bad", title);

// ---------- 覆盖层：模态框 / 抽屉，带焦点陷阱与 Esc ----------
const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';

function overlay(kind, { title, body, foot = "", width, onClose, labelIcon = "" }) {
  const root = document.getElementById("overlay-root");
  const prev = document.activeElement;
  const id = uid("dlg");
  const scrim = document.createElement("div");
  scrim.className = "scrim";
  const box = document.createElement("div");
  box.className = kind;
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-modal", "true");
  box.setAttribute("aria-labelledby", id);
  if (width) box.style.setProperty(kind === "drawer" ? "--drawer-w" : "width", width);
  const head = kind === "modal" ? "modal-head" : "drawer-head";
  box.innerHTML = `<div class="${head}"><h3 id="${id}" class="row">${labelIcon ? icon(labelIcon) : ""}${esc(title)}</h3>
      <button class="icon-btn sm" type="button" data-close aria-label="关闭">${icon("x", "sm")}</button></div>
    <div class="${kind}-body"></div>${foot ? `<div class="${kind}-foot"></div>` : ""}`;
  const bodyEl = box.querySelector(`.${kind}-body`);
  if (typeof body === "string") bodyEl.innerHTML = body; else if (body) bodyEl.appendChild(body);
  const footEl = box.querySelector(`.${kind}-foot`);
  if (footEl) { if (typeof foot === "string") footEl.innerHTML = foot; else footEl.appendChild(foot); }
  let closed = false;
  const close = (result) => {
    if (closed) return;
    closed = true;
    document.removeEventListener("keydown", onKey, true);
    scrim.remove(); box.remove();
    if (prev && prev.focus) prev.focus();
    onClose?.(result);
  };
  const onKey = (e) => {
    if (e.key === "Escape") { e.preventDefault(); close(); return; }
    if (e.key !== "Tab") return;
    const f = [...box.querySelectorAll(FOCUSABLE)].filter((x) => x.offsetParent !== null);
    if (!f.length) return;
    const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  };
  document.addEventListener("keydown", onKey, true);
  scrim.onclick = () => close();
  box.querySelectorAll("[data-close]").forEach((b) => (b.onclick = () => close()));
  root.append(scrim, box);
  const target = box.querySelector("[autofocus]") || box.querySelector(`.${kind}-body ${FOCUSABLE}`) || box.querySelector(FOCUSABLE);
  requestAnimationFrame(() => target?.focus());
  return { el: box, body: bodyEl, foot: footEl, close };
}
export const modal = (opts) => overlay("modal", opts);
export const drawer = (opts) => overlay("drawer", opts);

/** 简单确认框，返回 Promise<boolean> */
export function confirmDialog({ title, message, okText = "确认", danger = false }) {
  return new Promise((resolve) => {
    let ok = false;
    const m = modal({
      title, labelIcon: danger ? "alert" : "info",
      body: `<p>${esc(message)}</p>`,
      foot: `<button class="btn" type="button" data-close>取消</button><button class="btn ${danger ? "btn-danger" : "btn-primary"}" type="button" data-ok>${esc(okText)}</button>`,
      onClose: () => resolve(ok),
    });
    m.foot.querySelector("[data-close]").onclick = () => m.close();
    m.foot.querySelector("[data-ok]").onclick = () => { ok = true; m.close(); };
  });
}

/** 按钮进入加载态，返回恢复函数 */
export function busy(btn, text) {
  if (!btn) return () => {};
  const html = btn.innerHTML;
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  btn.innerHTML = `${icon("loader", "sm spin")}${text ? esc(text) : html}`;
  return () => { btn.disabled = false; btn.removeAttribute("aria-busy"); btn.innerHTML = html; };
}

/** 可访问的标签页：items=[{id,label,icon}]，返回 {select(id), badge(id, html)} */
export function tabs(container, items, onSelect, initial) {
  const name = uid("tab");
  container.setAttribute("role", "tablist");
  container.classList.add("tabs");
  container.innerHTML = items.map((t) => `<button role="tab" type="button" id="${name}-${t.id}" data-tab="${t.id}" aria-selected="false" tabindex="-1">${t.icon ? icon(t.icon, "sm") : ""}${esc(t.label)}<span data-badge="${t.id}"></span></button>`).join("");
  const btns = $$('[role="tab"]', container);
  const select = (id, focus = false) => {
    btns.forEach((b) => { const on = b.dataset.tab === id; b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus(); });
    onSelect(id);
  };
  btns.forEach((b, i) => {
    b.onclick = () => select(b.dataset.tab);
    b.onkeydown = (e) => {
      const d = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
      if (d) { e.preventDefault(); select(btns[(i + d + btns.length) % btns.length].dataset.tab, true); }
    };
  });
  select(initial || items[0].id);
  return { select, badge: (id, html) => { const el = container.querySelector(`[data-badge="${id}"]`); if (el) el.innerHTML = html; } };
}

/** 下载 JSON 文件 */
export function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/** 选择并读取一个 JSON 文件 */
export function pickJSON() {
  return new Promise((resolve, reject) => {
    const inp = document.createElement("input");
    inp.type = "file";
    inp.accept = "application/json,.json";
    inp.onchange = async () => {
      try { resolve(JSON.parse(await inp.files[0].text())); } catch { reject(new Error("文件不是合法的 JSON")); }
    };
    inp.click();
  });
}

export const debounce = (fn, wait = 200) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), wait); }; };
