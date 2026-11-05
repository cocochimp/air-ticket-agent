// 入口：顶栏（模式切换、用户选择、检索徽章、主题）+ 路由
import { hydrateIcons, icon } from "./icons.js";
import { startRouter } from "./router.js";
import { state, on, setUser, setMode, refreshStatus, loadUsers } from "./store.js";
import { esc, toast, toastError } from "./ui.js";

hydrateIcons();

// ---------- 主题 ----------
const themeBtn = document.getElementById("theme-toggle");
function paintTheme() {
  const dark = document.documentElement.dataset.theme !== "light";
  themeBtn.innerHTML = icon(dark ? "sun" : "moon");
  themeBtn.setAttribute("aria-label", dark ? "切换到浅色主题" : "切换到深色主题");
  themeBtn.title = themeBtn.getAttribute("aria-label");
}
themeBtn.onclick = () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch { /* 隐私模式 */ }
  paintTheme();
};
paintTheme();

// ---------- 模式切换 ----------
const modeEl = document.getElementById("mode-switch");
const MODES = [{ id: "offline", label: "规则模式", icon: "shield" }, { id: "deepseek", label: "DeepSeek", icon: "brain" }];
function paintMode() {
  modeEl.innerHTML = MODES.map((m) => `<button type="button" role="radio" aria-checked="${state.mode === m.id}" data-mode="${m.id}"
    title="${m.id === "deepseek" && !state.status?.deepseek_configured ? "未配置 API Key，请到配置中心填写" : m.label}">${icon(m.icon, "sm")}${m.label}${m.id === "deepseek" && state.status && !state.status.deepseek_configured ? icon("alert", "sm") : ""}</button>`).join("");
  modeEl.querySelectorAll("button").forEach((b, i, all) => {
    b.tabIndex = state.mode === b.dataset.mode ? 0 : -1;
    b.onclick = () => chooseMode(b.dataset.mode);
    b.onkeydown = (e) => {
      if (["ArrowLeft", "ArrowRight"].includes(e.key)) { e.preventDefault(); const n = all[(i + 1) % all.length]; n.focus(); chooseMode(n.dataset.mode); }
    };
  });
}
async function chooseMode(mode) {
  if (mode === state.mode) return;
  if (mode === "deepseek" && !state.status?.deepseek_configured) {
    toast("DeepSeek 模式需要先在配置中心填写 API Key", "warn", "尚未配置");
    location.hash = "#/config";
    return;
  }
  try { await setMode(mode); toast(`已切换为${mode === "offline" ? "规则模式" : "DeepSeek 模式"}`, "ok"); }
  catch (e) { toastError(e, "切换模式失败"); }
}

// ---------- 用户选择 ----------
const userSel = document.getElementById("user-select");
function paintUsers() {
  userSel.innerHTML = state.users.map((u) => `<option value="${esc(u.user_id)}" ${u.user_id === state.userId ? "selected" : ""}>${esc(u.name)} · ${esc(u.tier)}</option>`).join("");
}
userSel.onchange = () => setUser(userSel.value);

// ---------- 检索徽章 ----------
const badge = document.getElementById("retrieval-badge");
function paintBadge() {
  const r = state.status?.retrieval;
  if (!r) { badge.className = "badge"; badge.textContent = "检索 …"; return; }
  if (r.backend === "hybrid") {
    badge.className = "badge ok";
    badge.innerHTML = `${icon("layers", "sm")}检索 hybrid`;
    badge.title = "向量 + BM25 混合检索，精排模型已加载";
  } else {
    badge.className = "badge warn";
    badge.innerHTML = `${icon("alert", "sm")}检索 ${r.degraded ? "降级" : "fallback"}`;
    badge.title = r.degraded ? `检索模型不可用，已降级为字符 n-gram：${r.reason}` : "已配置为 fallback（字符 n-gram + BM25）";
  }
}

on((key) => {
  if (key === "users" || key === "user") paintUsers();
  if (key === "status" || key === "mode") { paintMode(); paintBadge(); }
});

paintMode();
Promise.all([loadUsers(), refreshStatus()]).catch((e) => toastError(e, "初始化失败"));
startRouter();
