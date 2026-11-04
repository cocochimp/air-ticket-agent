"""工作二：订单字段对齐、意图与多诉求拆解、指代消解、日期换算、Step-back、HyDE。"""
import re
from datetime import date, timedelta
from . import data, engine, seed
from . import settings as S
from .state import INTENT_LABEL, ParsedQuery, SubQuery

WEEK = "一二三四五六日"
DATE_PHRASE = (r"\d{4}-\d{2}-\d{2}|大后天|后天|明天|今天|下下(?:周|星期)[一二三四五六日天]|"
               r"下(?:周|星期)[一二三四五六日天]|(?:周|星期)[一二三四五六日天]|\d{1,2}月\d{1,2}[号日]?|\d{1,2}[号日]")
VAGUE_DATE = r"下个?月初|下个?月底|月初|月底|过几天|过两天|节后|年后"
INTENT_PATTERNS = [
    ("CHANGE", r"改签|改到|改成|换到|改期|改一下|换个时间|延后|往后挪|推迟|改天|换一天|换个日子|改日期|变更|改个时间"
               rf"|提前到(?:{DATE_PHRASE})"),
    ("REFUND", r"退票|退掉|退了|退款|不坐了|不去了|退多少|能退|退吗|怎么退|退一下|退钱"),
    ("STATUS", r"(?:航班|飞机|这班|那班).{0,6}(?:状态|延误|取消|准点|正常|情况|还飞)|(?:这票|这张|航班).{0,4}(?:啥|什么)情况"
               r"|还飞吗"),
    ("POLICY", r"规则|政策|规定"),
]
SPLIT = r"顺便|另外|还有|同时|以及|并且|再帮我|再看"
PRODUCT_SYNONYMS = [("SPECIAL", r"特价"), ("GROUP", r"团队|团购"), ("Y_DISCOUNT", r"打折|折扣"),
                    ("F", r"头等"), ("C", r"公务|商务舱"), ("Y", r"全价|经济舱")]
AIRLINE_SYNONYMS = {code: name for code, name in seed.AIRLINES.items()}
CABIN_LABEL = {"F": "头等舱", "C": "公务舱", "Y": "经济舱"}
SCENARIO_KIND = {"CHANGE": "VOLUNTARY_CHANGE", "REFUND": "VOLUNTARY_REFUND"}
KIND_LABEL = {"VOLUNTARY_CHANGE": "自愿改签", "VOLUNTARY_REFUND": "自愿退票", "CANCEL": "航班取消",
              "DELAY": "航班延误", "NO_SHOW": "误机", "FARE_DIFF": "改签差价"}


def today() -> date:
    return S.now().date()


def _next_month_day(month: int, day: int, d0: date) -> str | None:
    for y in range(d0.year, d0.year + 5):      # 2 月 29 日要等到下一个闰年
        try:
            c = date(y, month, day)
        except ValueError:
            continue
        if c >= d0:
            return c.isoformat()
    return None


def _next_day_of_month(day: int, d0: date) -> str | None:
    y, m = d0.year, d0.month
    for _ in range(14):                         # 当月没有这一天（如 9 月 31 日）就顺延到下个有的月份
        try:
            c = date(y, m, day)
        except ValueError:
            c = None
        if c and c >= d0:
            return c.isoformat()
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return None


def resolve_date(text: str) -> str | None:
    """把口语日期换算成 YYYY-MM-DD；认不出或日期不存在返回 None，绝不抛异常。
    周几按周一为一周起点：下周X = 下个自然周的周X，下下周X 再加一周；单独的周X 取今天之后最近的一个。"""
    t, d0 = text.strip(), today()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return date.fromisoformat(t).isoformat()
        except ValueError:
            return None
    rel = {"今天": 0, "明天": 1, "后天": 2, "大后天": 3}
    if t in rel:
        return (d0 + timedelta(days=rel[t])).isoformat()
    m = re.fullmatch(r"(下下|下)?(?:周|星期)([一二三四五六日天])", t)
    if m:
        wd = WEEK.index(m.group(2).replace("天", "日"))
        if m.group(1):
            weeks = 2 if m.group(1) == "下下" else 1
            return (d0 + timedelta(days=7 * weeks - d0.weekday() + wd)).isoformat()
        delta = wd - d0.weekday()
        return (d0 + timedelta(days=delta if delta > 0 else delta + 7)).isoformat()
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})[号日]?", t)
    if m:
        return _next_month_day(int(m.group(1)), int(m.group(2)), d0)
    m = re.fullmatch(r"(\d{1,2})[号日]", t)
    if m:
        return _next_day_of_month(int(m.group(1)), d0)
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
    prod = seed.PRODUCTS.get(o["fare_product"], ("", o["fare_product"]))[1]
    if not f:       # 航班记录缺失时仍给出可读标签
        return f"{o['flight_no']} {o['date']} {prod}"
    return f"{o['flight_no']} {f['origin_city']}→{f['dest_city']} {o['date']} {f['dep_time']} {prod}"


def _issued_orders(user_id):
    return [o for o in data.list_orders(user_id) if o["status"] == "ISSUED"]


def _future_orders(user_id):
    now = S.now()
    out = []
    for o in _issued_orders(user_id):
        f = data.get_flight(o["flight_no"], o["date"])
        if f and engine.departure(f) > now:
            out.append(o)
    return out


def _order_option(o: dict) -> dict:
    f = data.get_flight(o["flight_no"], o["date"])
    departed = bool(f) and engine.departure(f) <= S.now()
    return {"label": order_label(o) + ("（已起飞）" if departed else ""), "value": o["order_id"]}


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
            if o["status"] != "ISSUED" or o["order_id"] == anchor or not a:
                continue
            f = data.get_flight(o["flight_no"], o["date"])
            if f and (f["origin"], f["dest"]) == (a["dest"], a["origin"]):
                return o["order_id"]
    if re.search(r"上一张|之前那张|刚才那张|另一张", text) and len(stack) > 1:
        return stack[1]
    for m in re.finditer(r"(?<![A-Z0-9])[A-Z0-9]{2}\d{3,4}(?!\d)", text):
        for o in orders:
            if o["flight_no"] == m.group(0):
                return o["order_id"]
    for o in orders:
        f = data.get_flight(o["flight_no"], o["date"])
        if not f:
            continue
        o_c, d_c = re.escape(f["origin_city"]), re.escape(f["dest_city"])
        if re.search(f"去{d_c}|{o_c}飞{d_c}|{o_c}到{d_c}", text):
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
    if not f:
        return None
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
    for s in subs:   # 改签日期可能写在别的分句里；forced_date 是本轮追问刚拿到的值（图里每轮重置），优先于文本
        if s.intent == "CHANGE":
            s.target_date = ctx.get("forced_date") or s.target_date or find_date(question)

    p = ParsedQuery(intents=[s.intent for s in subs], sub_queries=subs)
    oid = resolve_order(question, ctx)
    if oid:
        o = data.get_order(oid)
        p.order_id, p.airline, p.cabin, p.fare_product = oid, o["airline"], o["cabin"], o["fare_product"]
        p.scenario = _flight_scenario(o)
    else:
        for code, name in AIRLINE_SYNONYMS.items():     # \b 在中文旁边不生效，改用字母数字边界
            if name in question or re.search(rf"(?<![A-Za-z0-9]){code}(?![A-Za-z0-9])", question):
                p.airline = code
                break
        for prod, pat in PRODUCT_SYNONYMS:
            if re.search(pat, question):
                p.fare_product, p.cabin = prod, seed.PRODUCTS[prod][0]
                break
        # 没有订单但说了票种：用户确实没有可办理的订单（或本来就在问政策）才按政策咨询处理；
        # 否则保留改签/退票意图，由缺 order_id 触发订单追问
        policy_only = all(s.intent == "POLICY" for s in subs)
        if (p.fare_product or p.airline) and (policy_only or not _issued_orders(ctx["user_id"])):
            first = subs[0].intent
            p.scenario = SCENARIO_KIND.get(first)
            p.sub_queries = [SubQuery(intent="POLICY", text=question)]
            p.intents = ["POLICY"]
    if not p.order_id and not p.scenario:   # 有订单时以航班状态为准，关键词只在没有订单时兜底
        if re.search(r"取消了|被取消", question):
            p.scenario = "CANCEL"
        elif re.search(r"延误", question):
            p.scenario = "DELAY"

    need_order = any(s.intent in ("CHANGE", "REFUND", "STATUS") for s in p.sub_queries)
    if need_order and not p.order_id:
        p.missing.append("order_id")
    if any(s.intent == "CHANGE" and not s.target_date for s in p.sub_queries):
        p.missing.append("target_date")
    p.confidence = round(max(0.0, 1.0 - 0.3 * ("order_id" in p.missing) - 0.2 * ("target_date" in p.missing)
                             - 0.5 * guessed), 2)

    if "order_id" in p.missing:
        opts = [_order_option(o) for o in _issued_orders(ctx["user_id"])]   # 已起飞的订单也可能要退（误机）
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
