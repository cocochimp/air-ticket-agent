"""工作二：订单字段对齐、意图与多诉求拆解、指代消解、日期换算、Step-back、HyDE。"""
import re
from datetime import date, timedelta
from . import data, engine, seed
from . import settings as S
from .state import INTENT_LABEL, ParsedQuery, SubQuery

INTENT_PATTERNS = [
    ("CHANGE", r"改签|改到|改成|换到|改期|改一下|换个时间|延后|提前到|往后挪"),
    ("REFUND", r"退票|退掉|退了|退款|不坐了|不去了|退多少|能退"),
    ("STATUS", r"航班状态|延误了吗|取消了吗|正常起飞|准点|啥情况|什么情况|还飞吗"),
    ("POLICY", r"规则|政策|规定"),
]
SPLIT = r"顺便|另外|还有|同时|以及|并且|再帮我|再看"
WEEK = "一二三四五六日"
DATE_PHRASE = (r"\d{4}-\d{2}-\d{2}|大后天|后天|明天|今天|下(?:周|星期)[一二三四五六日天]|"
               r"(?:周|星期)[一二三四五六日天]|\d{1,2}月\d{1,2}[号日]?|\d{1,2}[号日]")
VAGUE_DATE = r"下个?月初|下个?月底|月初|月底|过几天|过两天|节后|年后|下下周[一二三四五六日天]?"
PRODUCT_SYNONYMS = [("SPECIAL", r"特价"), ("GROUP", r"团队|团购"), ("Y_DISCOUNT", r"打折|折扣"),
                    ("F", r"头等"), ("C", r"公务|商务舱"), ("Y", r"全价|经济舱")]
AIRLINE_SYNONYMS = {code: name for code, name in seed.AIRLINES.items()}
CABIN_LABEL = {"F": "头等舱", "C": "公务舱", "Y": "经济舱"}
SCENARIO_KIND = {"CHANGE": "VOLUNTARY_CHANGE", "REFUND": "VOLUNTARY_REFUND"}
KIND_LABEL = {"VOLUNTARY_CHANGE": "自愿改签", "VOLUNTARY_REFUND": "自愿退票", "CANCEL": "航班取消",
              "DELAY": "航班延误", "NO_SHOW": "误机", "FARE_DIFF": "改签差价"}


def today() -> date:
    return S.now().date()


def resolve_date(text: str) -> str | None:
    t, d0 = text.strip(), today()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return t
    rel = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}
    if t in rel:
        return (d0 + timedelta(days=rel[t])).isoformat()
    m = re.fullmatch(r"(下)?(?:周|星期)([一二三四五六日天])", t)
    if m:
        wd = WEEK.index(m.group(2).replace("天", "日"))
        if m.group(1):
            return (d0 + timedelta(days=7 - d0.weekday() + wd)).isoformat()
        delta = wd - d0.weekday()
        return (d0 + timedelta(days=delta if delta > 0 else delta + 7)).isoformat()
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})[号日]?", t)
    if m:
        y = d0.year if (int(m.group(1)), int(m.group(2))) >= (d0.month, d0.day) else d0.year + 1
        return date(y, int(m.group(1)), int(m.group(2))).isoformat()
    m = re.fullmatch(r"(\d{1,2})[号日]", t)
    if m:
        day = int(m.group(1))
        y, mo = (d0.year, d0.month) if day >= d0.day else ((d0.year + 1, 1) if d0.month == 12 else (d0.year, d0.month + 1))
        return date(y, mo, day).isoformat()
    return None


def find_date(text: str):
    """返回 (标准日期 | 原始模糊短语 | None)。"""
    m = re.search(DATE_PHRASE, text)
    if m:
        return resolve_date(m.group(0)) or m.group(0)
    m = re.search(VAGUE_DATE, text)
    return m.group(0) if m else None


def detect_intents(text: str) -> list[str]:
    hits = []
    for intent, pat in INTENT_PATTERNS:
        m = re.search(pat, text)
        if m:
            hits.append((m.start(), intent))
    return [i for _, i in sorted(hits)]


def order_label(o: dict) -> str:
    f = data.get_flight(o["flight_no"], o["date"])
    return (f"{o['flight_no']} {f['origin_city']}→{f['dest_city']} {o['date']} {f['dep_time']} "
            f"{seed.PRODUCTS[o['fare_product']][1]}")


def _future_orders(user_id):
    now = S.now()
    return [o for o in data.list_orders(user_id) if o["status"] == "ISSUED"
            and engine.departure(data.get_flight(o["flight_no"], o["date"])) > now]


def resolve_order(text: str, ctx: dict) -> str | None:
    orders = data.list_orders(ctx["user_id"])
    by_id = {o["order_id"]: o for o in orders}
    m = re.search(r"ORD\d{11}", text)
    if m and m.group(0) in by_id:
        return m.group(0)
    stack = [x for x in ctx.get("order_stack", []) if x in by_id]
    anchor = ctx.get("selected_order_id") or (stack[0] if stack else None)
    if re.search(r"回程|返程|回来那", text) and anchor:
        a = data.get_flight(by_id[anchor]["flight_no"], by_id[anchor]["date"])
        for o in orders:
            f = data.get_flight(o["flight_no"], o["date"])
            if (f["origin"], f["dest"]) == (a["dest"], a["origin"]) and o["order_id"] != anchor:
                return o["order_id"]
    if re.search(r"上一张|之前那张|刚才那张|另一张", text) and len(stack) > 1:
        return stack[1]
    m = re.search(r"[A-Z0-9]{2}\d{3,4}", text)
    if m:
        for o in orders:
            if o["flight_no"] == m.group(0):
                return o["order_id"]
    for o in orders:
        f = data.get_flight(o["flight_no"], o["date"])
        if re.search(f"去{f['dest_city']}|{f['origin_city']}飞{f['dest_city']}|{f['origin_city']}到{f['dest_city']}", text):
            return o["order_id"]
    if ctx.get("selected_order_id") in by_id:
        return ctx["selected_order_id"]
    if stack and re.search(r"它|这张|这个|那张|这班", text):
        return stack[0]
    future = _future_orders(ctx["user_id"])
    if len(future) == 1:
        return future[0]["order_id"]
    return stack[0] if stack else None


def _flight_scenario(order) -> str | None:
    f = data.get_flight(order["flight_no"], order["date"])
    inv = engine.involuntary_reason(f)
    if inv:
        return {"GEN-CANCEL": "CANCEL", "GEN-DELAY": "DELAY"}[inv]
    if engine.departure(f) <= S.now():
        return "NO_SHOW"
    return None


def date_options(n=5):
    d0 = today()
    return [{"label": f"{(d0 + timedelta(days=i)).isoformat()}（{'周' + WEEK[(d0 + timedelta(days=i)).weekday()]}）",
             "value": (d0 + timedelta(days=i)).isoformat()} for i in range(1, n + 1)]


def understand(question: str, ctx: dict, llm=None) -> ParsedQuery:
    segments = [s for s in re.split(SPLIT, question) if s.strip()] or [question]
    subs, guessed = [], False
    for seg in segments:
        for intent in detect_intents(seg):
            if not any(s.intent == intent for s in subs):
                subs.append(SubQuery(intent=intent, text=seg.strip(" ，,。"),
                                     target_date=find_date(seg) if intent == "CHANGE" else None))
    if llm is not None:
        subs = llm.parse_intents(question, ctx, subs) or subs
    if not subs:
        intent = ctx.get("forced_intent") or ctx.get("last_intent") or "POLICY"
        guessed = not (ctx.get("forced_intent") or ctx.get("last_intent"))
        subs = [SubQuery(intent=intent, text=question, target_date=find_date(question) if intent == "CHANGE" else None)]
    for s in subs:   # 改签日期可能写在别的分句里；追问得到的日期优先
        if s.intent == "CHANGE":
            s.target_date = ctx.get("forced_date") or s.target_date or find_date(question)

    p = ParsedQuery(intents=[s.intent for s in subs], sub_queries=subs)
    oid = resolve_order(question, ctx)
    if oid:
        o = data.get_order(oid)
        p.order_id, p.airline, p.cabin, p.fare_product = oid, o["airline"], o["cabin"], o["fare_product"]
        p.scenario = _flight_scenario(o)
    else:
        for code, name in AIRLINE_SYNONYMS.items():
            if name in question or re.search(rf"\b{code}\b", question):
                p.airline = code
        for prod, pat in PRODUCT_SYNONYMS:
            if re.search(pat, question):
                p.fare_product, p.cabin = prod, seed.PRODUCTS[prod][0]
                break
        if p.fare_product or p.airline:   # 没有订单但说了票种：按政策咨询处理
            first = subs[0].intent
            p.scenario = SCENARIO_KIND.get(first)
            p.sub_queries = [SubQuery(intent="POLICY", text=question)]
            p.intents = ["POLICY"]
    if re.search(r"取消了|被取消", question) and not p.scenario:
        p.scenario = "CANCEL"
    if re.search(r"延误", question) and not p.scenario:
        p.scenario = "DELAY"

    need_order = any(s.intent in ("CHANGE", "REFUND", "STATUS") for s in p.sub_queries)
    if need_order and not p.order_id:
        p.missing.append("order_id")
    if any(s.intent == "CHANGE" and not s.target_date for s in p.sub_queries):
        p.missing.append("target_date")
    p.confidence = round(max(0.0, 1.0 - 0.3 * ("order_id" in p.missing) - 0.2 * ("target_date" in p.missing)
                             - 0.3 * guessed), 2)

    if "order_id" in p.missing:
        opts = [{"label": order_label(o), "value": o["order_id"]} for o in _future_orders(ctx["user_id"])]
        p.clarify = {"field": "order_id", "question": "您想办理哪一张订单？", "options": opts}
    elif "target_date" in p.missing:
        p.clarify = {"field": "target_date", "question": "想改签到哪一天？", "options": date_options()}
    elif p.confidence < S.get()["query"]["clarify_threshold"]:
        p.clarify = {"field": "intent", "question": "您是想改签、退票，还是了解规则？",
                     "options": [{"label": v, "value": k} for k, v in INTENT_LABEL.items()]}

    desc = f"订单{p.order_id}（{order_label(data.get_order(p.order_id))}）" if p.order_id else "未指定订单"
    asks = "；".join(INTENT_LABEL[s.intent] + (f"至{s.target_date}" if s.target_date else "") for s in p.sub_queries)
    p.rewritten = f"{desc}：{asks}"
    return p


def scenario_for(p: ParsedQuery, sub: SubQuery) -> str | None:
    if p.scenario in ("CANCEL", "DELAY", "NO_SHOW"):
        return p.scenario
    return SCENARIO_KIND.get(sub.intent) or p.scenario


def step_back_query(p: ParsedQuery, sub: SubQuery) -> str:
    cabin = CABIN_LABEL.get(p.cabin or "", "")
    return f"{cabin}{KIND_LABEL.get(scenario_for(p, sub) or '', '退改')}规则"


def hyde_text(p: ParsedQuery, sub: SubQuery) -> str:
    name = seed.AIRLINES.get(p.airline or "", "")
    prod = seed.PRODUCTS.get(p.fare_product or "", ("", ""))[1]
    kind = KIND_LABEL.get(scenario_for(p, sub) or "", "退改")
    verb = "改期" if sub.intent == "CHANGE" else "退票"
    return f"{name}{prod}{kind}规则：起飞前不同时间申请{kind}，按票面价一定比例收取{verb}费。"
