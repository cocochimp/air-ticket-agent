// 检索漏斗渲染器：对话页与规则库检索调试器共用。
// funnel = [{stage,label,k,count,items:[{id,title,score}]}]
import { icon } from "./icons.js";
import { esc, empty } from "./ui.js";

const STAGE_ICON = { dense: "brain", sparse: "search", rrf: "git", filter: "filter", rerank: "bar", mmr: "layers", inject: "zap" };
const STAGE_HELP = {
  dense: "向量相似度召回 top-k", sparse: "BM25 关键词召回 top-k", rrf: "两路结果按排名倒数融合",
  filter: "按航司/舱位/票种硬过滤 + 相关性门槛", rerank: "交叉编码器精排，低于 rerank_min 截断，画像加权",
  mmr: "最大边际相关去冗余", inject: "最终注入生成上下文",
};

/**
 * @param container DOM
 * @param funnel 漏斗数据
 * @param opts {highlight: rule_id, initial: stage, meta: html}
 * @returns {{highlight(id)}}
 */
export function renderFunnel(container, funnel, opts = {}) {
  if (!funnel || !funnel.length) {
    container.innerHTML = empty("暂无检索数据", opts.emptyHint || "触发一次规则检索后，这里会显示每层漏斗", "funnel");
    return { highlight() {} };
  }
  const max = Math.max(1, ...funnel.map((s) => Math.max(s.count, s.k || 0)));
  let active = opts.initial || (opts.highlight ? stageOf(funnel, opts.highlight) : null) || funnel[funnel.length - 1].stage;
  let hl = opts.highlight || null;

  const draw = () => {
    const s = funnel.find((x) => x.stage === active) || funnel[0];
    container.innerHTML = `${opts.meta || ""}
      <div class="funnel" role="list" aria-label="检索漏斗">
        ${funnel.map((st) => {
          const w = Math.max(2, (st.count / max) * 100);
          const hasHl = hl && st.items.some((x) => x.id === hl);
          return `<button type="button" role="listitem" class="fn-row ${st.stage === active ? "active" : ""}" data-stage="${st.stage}" aria-pressed="${st.stage === active}" title="${esc(STAGE_HELP[st.stage] || "")}">
            <span class="fn-label">${icon(STAGE_ICON[st.stage] || "circle", "sm")}${esc(st.label)}</span>
            <span class="fn-track"><span class="fn-bar fn-${st.stage}" style="width:${w}%"></span>${st.k ? `<span class="fn-k" style="left:${Math.min(100, (st.k / max) * 100)}%" title="k=${st.k}"></span>` : ""}</span>
            <span class="fn-count num">${st.count}${st.k ? `<span class="faint">/${st.k}</span>` : ""}</span>
            ${hasHl ? `<span class="fn-hl" title="包含高亮规则">${icon("target", "sm")}</span>` : `<span class="fn-hl"></span>`}
          </button>`;
        }).join("")}
      </div>
      <div class="fn-detail">
        <div class="row between"><h4 class="row">${icon(STAGE_ICON[s.stage] || "circle", "sm")}${esc(s.label)} <span class="faint" style="font-weight:400">${esc(STAGE_HELP[s.stage] || "")}</span></h4><span class="num muted">${s.items.length} 条</span></div>
        ${s.items.length ? `<ol class="fn-items">${s.items.map((x, i) => {
          const sc = typeof x.score === "number" ? x.score : 0;
          const norm = s.stage === "rrf" ? Math.min(1, sc * 30) : s.stage === "sparse" ? Math.min(1, sc / 10) : Math.max(0, Math.min(1, sc));
          return `<li class="${x.id === hl ? "hl" : ""}" data-rule="${esc(x.id)}">
            <span class="num faint fn-rank">${i + 1}</span>
            <span class="fn-doc"><code class="fn-id">${esc(x.id)}</code><span class="fn-title">${esc(x.title)}</span></span>
            <span class="fn-score"><span class="fn-sbar"><span style="width:${(norm * 100).toFixed(0)}%"></span></span><span class="num">${sc.toFixed(4)}</span></span>
          </li>`;
        }).join("")}</ol>` : empty("该层没有文档", "被上一层的过滤或门槛截断", "filter")}
      </div>`;
    container.querySelectorAll(".fn-row").forEach((b) => (b.onclick = () => { active = b.dataset.stage; draw(); }));
    const el = container.querySelector(".fn-items li.hl");
    if (el) el.scrollIntoView({ block: "nearest" });
  };
  draw();
  return {
    highlight(id) { hl = id; active = stageOf(funnel, id) || active; draw(); },
  };
}

/** 找到包含该规则的最后一层（越靠后越能说明为何入选/落选） */
function stageOf(funnel, id) {
  const hit = [...funnel].reverse().find((s) => s.items.some((x) => x.id === id));
  return hit ? hit.stage : null;
}
