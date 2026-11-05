// Hash 路由：#/chat #/eval #/rules #/data #/config。每个页面模块导出 mount(el, params) → 可选 unmount 函数。
import { icon } from "./icons.js";
import { toastError } from "./ui.js";

export const ROUTES = [
  { path: "chat", label: "对话", title: "对话工作台", icon: "chat", load: () => import("./pages/chat.js") },
  { path: "eval", label: "评测", title: "评测中心", icon: "target", load: () => import("./pages/eval.js") },
  { path: "rules", label: "规则", title: "规则库", icon: "book", load: () => import("./pages/rules.js") },
  { path: "data", label: "数据", title: "订单与用户", icon: "database", load: () => import("./pages/data.js") },
  { path: "config", label: "配置", title: "配置中心", icon: "settings", load: () => import("./pages/config.js") },
];

let cleanup = null;
let seq = 0;

export function renderNav() {
  document.getElementById("nav").innerHTML = ROUTES.map((r) => `<li><a href="#/${r.path}" data-path="${r.path}" title="${r.title}">${icon(r.icon)}<span>${r.label}</span></a></li>`).join("");
}

async function go() {
  const [path, query = ""] = location.hash.replace(/^#\/?/, "").split("?");
  const route = ROUTES.find((r) => r.path === path);
  if (!route) { location.replace("#/chat"); return; }
  document.querySelectorAll("#nav a").forEach((a) => (a.dataset.path === route.path ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  document.title = `${route.title} · 退改签助手`;
  const my = ++seq;
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
  const view = document.getElementById("view");
  view.innerHTML = `<div class="page"><div class="skeleton" style="height:28px;width:220px;margin-bottom:16px"></div><div class="skeleton" style="height:320px"></div></div>`;
  try {
    const mod = await route.load();
    if (my !== seq) return;
    view.innerHTML = "";
    view.scrollTop = 0;
    cleanup = (await mod.mount(view, new URLSearchParams(query))) || null;
  } catch (e) {
    console.error(e);
    toastError(e, "页面加载失败");
  }
}

export function startRouter() {
  renderNav();
  window.addEventListener("hashchange", go);
  go();
}
