// 订单与用户：订单表 + 航班状态切换（正常/延误/取消）、用户画像卡片（清空长期记忆）、重置演示数据
import { get, post, del } from "../api.js";
import { icon } from "../icons.js";
import { loadUsers } from "../store.js";
import { esc, empty, skeleton, modal, toast, toastError, confirmDialog, busy, money, AIRLINES, PRODUCTS, INTENT } from "../ui.js";

const ST = [["scheduled", "正常", "checkCircle"], ["delayed", "延误", "clock"], ["cancelled", "取消", "xCircle"]];

export async function mount(view) {
  view.innerHTML = `<div class="page">
    <div class="page-head"><div><h2>订单与用户</h2><p>演示数据均为虚构。修改航班状态后，对话中的退改费用会按非自愿规则重新计算。</p></div>
      <button class="btn btn-danger" type="button" data-reset>${icon("rotate", "sm")}重置演示数据</button></div>
    <section aria-labelledby="u-h"><h3 id="u-h" class="row" style="margin-bottom:12px">${icon("users", "sm")}用户画像</h3><div class="user-cards" data-users>${skeleton(2, 120)}</div></section>
    <section class="card" style="margin-top:20px" aria-labelledby="o-h">
      <div class="card-head"><h3 id="o-h">${icon("ticket", "sm")}订单 <span class="num muted" data-count></span></h3></div>
      <div class="table-wrap flat" data-orders>${skeleton(8, 24)}</div>
    </section></div>`;
  const $ = (s) => view.querySelector(s);
  let users = [], orders = [];

  const topList = (d, label = (k) => k) => Object.entries(d || {}).sort((a, b) => b[1] - a[1]).slice(0, 4);
  const bars = (d, label) => { const l = topList(d); const max = Math.max(1, ...l.map((x) => x[1])); return l.length ? `<ul class="hbars small">${l.map(([k, v]) => `<li><span class="hb-label">${esc(label(k))}</span><span class="hb-track"><span style="width:${(v / max) * 100}%"></span></span><span class="num hb-val">${v}</span></li>`).join("")}</ul>` : `<p class="faint">暂无</p>`; };
  const drawUsers = () => {
    $("[data-users]").innerHTML = users.map((u) => { const d = u.data || {}; const n = orders.filter((o) => o.user_id === u.user_id).length; return `<article class="card user-card">
      <header class="row between"><div class="row"><span class="u-avatar" aria-hidden="true">${esc(u.name.slice(0, 1))}</span><div><h4>${esc(u.name)}</h4><span class="faint"><code>${esc(u.user_id)}</code> · ${n} 个订单</span></div></div><span class="chip ${u.tier === "金卡" ? "warn" : u.tier === "银卡" ? "info" : ""}">${esc(u.tier)}</span></header>
      <div class="u-grid"><div><div class="mini-label">常走航线</div>${bars(d.routes, (k) => k)}</div><div><div class="mini-label">偏好航司</div>${bars(d.airlines, (k) => AIRLINES[k] || k)}</div>
      <div><div class="mini-label">常买票种</div>${bars(d.products, (k) => PRODUCTS[k] || k)}</div><div><div class="mini-label">退改偏好</div>${bars(d.prefer_action, (k) => INTENT[k] || k)}</div></div>
      <footer class="row between"><span class="faint">价格敏感度 <strong>${esc(d.price_sensitivity || "—")}</strong></span><button type="button" class="btn btn-sm" data-clear="${esc(u.user_id)}">${icon("eraser", "sm")}清空长期记忆</button></footer></article>`; }).join("") || empty("没有用户", "", "users");
    view.querySelectorAll("[data-clear]").forEach((b) => (b.onclick = async () => {
      const u = users.find((x) => x.user_id === b.dataset.clear);
      if (!(await confirmDialog({ title: "清空长期记忆", message: `清空 ${u.name} 的画像统计（航线、航司、票种、偏好）？会话短期记忆不受影响。`, okText: "清空", danger: true }))) return;
      try { await del(`/api/profiles/${encodeURIComponent(u.user_id)}/memory`); toast(`已清空 ${u.name} 的长期记忆`, "ok"); await refresh(); } catch (e) { toastError(e); }
    }));
  };
  const userName = (id) => users.find((u) => u.user_id === id)?.name || id;
  const drawOrders = () => {
    $("[data-count]").textContent = String(orders.length);
    if (!orders.length) { $("[data-orders]").innerHTML = empty("没有订单", "点击「重置演示数据」重新生成", "ticket"); return; }
    $("[data-orders]").innerHTML = `<table class="table"><thead><tr><th>订单号</th><th>乘客</th><th>航班</th><th>航线</th><th>日期 时间</th><th>票种</th><th class="num">票面价</th><th>订单状态</th><th>航班状态</th></tr></thead><tbody>
      ${orders.map((o, i) => { const f = o.flight || {}; return `<tr><td><code>${esc(o.order_id)}</code> <span class="faint num">#${i + 1}</span></td><td>${esc(userName(o.user_id))}</td>
        <td><span class="num">${esc(o.flight_no)}</span> <span class="faint">${esc(o.airline_name)}</span></td><td>${esc(f.origin_city)} ${icon("arrowRight", "sm")} ${esc(f.dest_city)}</td>
        <td class="num">${esc(o.date)} ${esc(f.dep_time || "")}</td><td>${esc(o.product_label)}</td><td class="num">${money(o.fare)}</td>
        <td><span class="chip ${o.status === "ISSUED" ? "" : o.status === "REFUNDED" ? "bad" : "info"}">${esc(o.status_label)}</span></td>
        <td><div class="status-switch" role="radiogroup" aria-label="${esc(o.flight_no)} ${esc(o.date)} 航班状态">
          ${ST.map(([v, l, ic]) => `<button type="button" role="radio" aria-checked="${o.flight_status === v}" class="ss-${v}" data-fs="${v}" data-i="${i}">${icon(ic, "sm")}${l}${v === "delayed" && o.flight_status === "delayed" ? ` <span class="num">${o.delay_min}′</span>` : ""}</button>`).join("")}
        </div></td></tr>`; }).join("")}</tbody></table>`;
    view.querySelectorAll("[data-fs]").forEach((b) => (b.onclick = () => setStatus(orders[+b.dataset.i], b.dataset.fs, b)));
  };
  const setStatus = async (o, status, btn) => {
    let delay = 0;
    if (status === "delayed") {
      delay = await askDelay(o);
      if (delay === null) return;
    }
    const done = busy(btn);
    try {
      await post(`/api/flights/${encodeURIComponent(o.flight_no)}/${encodeURIComponent(o.date)}/status`, { status, delay_min: delay });
      toast(`${o.flight_no} ${o.date} 已设为${ST.find((s) => s[0] === status)[1]}${delay ? ` ${delay} 分钟` : ""}`, "ok");
      await refresh();
    } catch (e) { done(); toastError(e, "修改失败"); }
  };
  /** 延误分钟数输入框（模态），取消返回 null */
  const askDelay = (o) => new Promise((resolve) => {
    let val = null;
    const m = modal({
      title: `${o.flight_no} ${o.date} 设为延误`, labelIcon: "clock",
      body: `<div class="field" data-field="delay"><label for="delay-min">延误分钟数 <span class="range">1–1440</span></label>
        <input class="input num" id="delay-min" type="number" min="1" max="1440" step="10" value="${o.delay_min || 150}" autofocus>
        <span class="help">延误 ≥ 120 分钟时按非自愿规则免费退改</span><span class="err" hidden></span></div>`,
      foot: `<button class="btn" type="button" data-close>取消</button><button class="btn btn-primary" type="button" data-ok>${icon("check", "sm")}确定</button>`,
      onClose: () => resolve(val),
    });
    const inp = m.body.querySelector("input");
    const ok = () => {
      const v = Number(inp.value);
      if (!(Number.isInteger(v) && v > 0 && v <= 1440)) {
        const f = m.body.querySelector(".field"); f.classList.add("invalid");
        const e = f.querySelector(".err"); e.hidden = false; e.textContent = "请输入 1–1440 之间的整数";
        inp.focus(); return;
      }
      val = v; m.close();
    };
    inp.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); ok(); } };
    m.foot.querySelector("[data-close]").onclick = () => m.close();
    m.foot.querySelector("[data-ok]").onclick = ok;
  });
  const refresh = async () => {
    try {
      const [u, o] = await Promise.all([get("/api/users"), get("/api/orders")]);
      users = u.users; orders = o.orders;
    } catch (e) { toastError(e, "加载失败"); }
    drawUsers(); drawOrders();
  };
  $("[data-reset]").onclick = async (e) => {
    if (!(await confirmDialog({ title: "重置演示数据", message: "将重新生成所有航班、订单、用户画像和规则，已办理的改签/退票与规则编辑都会丢失。", okText: "重置", danger: true }))) return;
    const done = busy(e.currentTarget, "重置中");
    try { await post("/api/data/reset"); toast("演示数据已重置", "ok"); await refresh(); await loadUsers(); } catch (err) { toastError(err, "重置失败"); } finally { done(); }
  };
  await refresh();
}
