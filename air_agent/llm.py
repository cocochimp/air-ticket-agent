"""DeepSeek（OpenAI 兼容）客户端与测试用 FakeLLM。任何异常都抛 LLMError，由调用方降级。"""
import json
import httpx
from pydantic import BaseModel, ValidationError
from . import settings as S
from .state import SchemeOutput, SubQuery


class LLMError(Exception):
    pass


class LLMNotConfigured(LLMError):
    pass


class _Intents(BaseModel):
    sub_queries: list[SubQuery]


class DeepSeekClient:
    def __init__(self, transport=None):
        cfg = S.get()["llm"]
        self.model = cfg["model"]
        self.http = httpx.Client(base_url=cfg["base_url"], timeout=cfg["timeout"], transport=transport,
                                 headers={"Authorization": f"Bearer {cfg['api_key']}"})
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, tools=None, json_mode=False) -> dict:
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = self.http.post("/chat/completions", json=body)
            r.raise_for_status()
            payload = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        u = payload.get("usage", {})
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        return payload["choices"][0]["message"]

    def _json(self, system, user) -> dict:
        msg = self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], json_mode=True)
        try:
            return json.loads(msg.get("content") or "{}")
        except json.JSONDecodeError as e:
            raise LLMError(f"JSON 解析失败：{e}") from e

    def react(self, messages, tools=None) -> dict:
        from .tools import llm_specs
        msg = self.chat(messages, tools=tools or llm_specs())
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"thought": msg.get("content") or "", "action": "finish", "args": {}}
        fn = calls[0]["function"]
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments")}
        return {"thought": msg.get("content") or "", "action": fn["name"], "args": args}

    def parse_intents(self, question, ctx, rule_subs):
        try:
            out = self._json(
                "把乘客的退改签诉求拆成子诉求，输出 JSON：{\"sub_queries\":[{\"intent\":\"CHANGE|REFUND|POLICY|STATUS\","
                "\"text\":\"子诉求原文\",\"target_date\":\"改签目标日期原文或null\"}]}。不要编造日期。",
                f"上一轮意图：{ctx.get('last_intent')}\n乘客：{question}")
            subs = _Intents.model_validate(out).sub_queries
        except (LLMError, ValidationError):
            return None
        from .query import resolve_date
        for s in subs:
            if s.target_date:
                s.target_date = resolve_date(s.target_date) or s.target_date
        return subs

    def summarize(self, turns):
        try:
            return self._json("把对话压缩成一句话摘要，保留订单、日期、金额。输出 {\"summary\": \"...\"}",
                              json.dumps(turns, ensure_ascii=False)).get("summary")
        except LLMError:
            return None

    def write_scheme(self, draft: SchemeOutput, results, feedback, context):
        prompt = (f"上下文：\n{json.dumps({k: v['text'] for k, v in context.get('blocks', {}).items()}, ensure_ascii=False)}\n"
                  f"方案草稿：{draft.model_dump_json()}\n"
                  f"上次校验的问题：{feedback}\n"
                  "请把 text 改写得更自然，金额必须与草稿逐字一致，不得新增规则里没有的承诺。输出 {\"text\": \"...\"}")
        try:
            out = self._json("你是航空公司退改签客服，只能依据给定规则和金额作答。", prompt)
            return SchemeOutput(options=draft.options, text=out["text"], texts=draft.texts)
        except (LLMError, KeyError, ValidationError):
            return None

    def fact_check(self, sentences, corpus):
        try:
            out = self._json("判断每句话能否在规则原文中找到依据。输出 {\"unsupported\": [\"无依据的句子\"]}",
                             f"规则原文：{corpus}\n句子：{json.dumps(sentences, ensure_ascii=False)}")
            return [s for s in out.get("unsupported", []) if s in sentences]
        except LLMError:
            return None


class FakeLLM:
    """按脚本回放 react()；其余方法返回 None，让调用方走规则逻辑。"""

    def __init__(self, react=None, scheme_texts=None):
        self.script = list(react or [])
        self.scheme_texts = list(scheme_texts or [])   # 每次 write_scheme 取一个；None 表示用模板
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def react(self, messages, tools=None):
        self.usage["calls"] += 1
        if not self.script:
            raise LLMError("脚本已用完")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def parse_intents(self, *a):
        return None

    def summarize(self, *a):
        return None

    def write_scheme(self, draft, *a):
        text = self.scheme_texts.pop(0) if self.scheme_texts else None
        return SchemeOutput(options=draft.options, text=text, texts=draft.texts) if text else None

    def fact_check(self, *a):
        return None


def get_client(mode: str):
    if mode != "deepseek":
        return None
    if not S.get()["llm"]["api_key"]:
        raise LLMNotConfigured("DeepSeek 模式需要先在配置中心填写 API Key")
    return DeepSeekClient()
