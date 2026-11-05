// 后端接口封装：JSON 请求 + SSE（fetch 流式读取，支持 POST）
export class ApiError extends Error {
  constructor(status, body) {
    const e = (body && body.error) || {};
    super(e.message || `请求失败（HTTP ${status}）`);
    this.status = status; this.code = e.code || "HTTP_ERROR"; this.fields = e.fields || {};
  }
}

export async function req(method, url, body) {
  let res;
  try {
    res = await fetch(url, {
      method, headers: body !== undefined ? { "Content-Type": "application/json" } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    throw new ApiError(0, { error: { code: "NETWORK", message: "无法连接服务，请确认后端已启动" } });
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

export const get = (u) => req("GET", u);
export const post = (u, b = {}) => req("POST", u, b);
export const put = (u, b) => req("PUT", u, b);
export const del = (u) => req("DELETE", u);

/**
 * 读取 SSE 流。handlers: {trace, interrupt, final, error, progress, done, log}
 * 返回 Promise，流结束时 resolve。signal 可中止。
 */
export async function stream(method, url, body, handlers = {}, signal) {
  let res;
  try {
    res = await fetch(url, {
      method, signal, headers: body ? { "Content-Type": "application/json", Accept: "text/event-stream" } : { Accept: "text/event-stream" },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    if (e.name === "AbortError") return;
    handlers.error?.({ code: "NETWORK", message: "连接中断，请重试" });
    return;
  }
  if (!res.ok) {
    let data = null;
    try { data = await res.json(); } catch { /* 非 JSON */ }
    const err = new ApiError(res.status, data);
    handlers.error?.({ code: err.code, message: err.message, fields: err.fields });
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  const dispatch = (block) => {
    let ev = "message", data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) ev = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (!data) return;
    let payload;
    try { payload = JSON.parse(data); } catch { payload = data; }
    handlers[ev]?.(payload);
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) { dispatch(buf.slice(0, i)); buf = buf.slice(i + 2); }
    }
    if (buf.trim()) dispatch(buf);
  } catch (e) {
    if (e.name !== "AbortError") handlers.error?.({ code: "NETWORK", message: "流式连接中断" });
  }
}
