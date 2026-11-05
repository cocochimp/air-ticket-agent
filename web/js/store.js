// 全局状态：当前用户、模式、检索状态。页面通过 on() 订阅变化。
import { get, put } from "./api.js";

const subs = new Set();
export const state = {
  users: [],
  userId: localStorage.getItem("user_id") || "u_zhang",
  mode: "offline",
  status: null,      // /api/status
};

export function on(fn) { subs.add(fn); return () => subs.delete(fn); }
function emit(key) { subs.forEach((fn) => { try { fn(key, state); } catch (e) { console.error(e); } }); }

export function setUser(id) {
  if (id === state.userId) return;
  state.userId = id;
  localStorage.setItem("user_id", id);
  emit("user");
}

export async function setMode(mode) {
  await put("/api/config", { mode });
  state.mode = mode;
  emit("mode");
  await refreshStatus();
}

export async function refreshStatus() {
  state.status = await get("/api/status");
  state.mode = state.status.mode;
  emit("status");
  return state.status;
}

export async function loadUsers() {
  state.users = (await get("/api/users")).users;
  if (!state.users.some((u) => u.user_id === state.userId) && state.users[0]) state.userId = state.users[0].user_id;
  emit("users");
  return state.users;
}

export const currentUser = () => state.users.find((u) => u.user_id === state.userId);
