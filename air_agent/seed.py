"""确定性生成虚构演示数据：航司、航班、订单、用户画像、费率表与退改规则。"""
from datetime import datetime, timedelta

AIRLINES = {"MU": "东方云航", "CA": "国际星航", "CZ": "南方海航", "HU": "海岛航空"}
# fare_product -> (舱位大类, 中文名)
PRODUCTS = {"F": ("F", "头等舱全价"), "C": ("C", "公务舱全价"), "Y": ("Y", "经济舱全价"),
            "Y_DISCOUNT": ("Y", "经济舱折扣"), "SPECIAL": ("Y", "经济舱特价"), "GROUP": ("Y", "团队票")}
PRODUCT_FACTOR = {"F": 3.0, "C": 2.0, "Y": 1.0, "Y_DISCOUNT": 0.6, "SPECIAL": 0.35, "GROUP": 0.5}
WINDOWS = ["d7", "h48", "h4", "lt4", "after"]
WINDOW_LABEL = {"d7": "起飞前7天(含)以上", "h48": "起飞前48小时(含)至7天",
                "h4": "起飞前4小时(含)至48小时", "lt4": "起飞前4小时以内", "after": "起飞后"}
# 费率（票面价百分比），按 WINDOWS 顺序；None 表示不允许
BASE_RATES = {
    "F": {"change": [0, 0, 5, 10, 20], "refund": [0, 5, 10, 15, 25]},
    "C": {"change": [0, 5, 10, 15, 25], "refund": [5, 10, 15, 20, 30]},
    "Y": {"change": [5, 10, 15, 20, 30], "refund": [10, 15, 20, 25, 40]},
    "Y_DISCOUNT": {"change": [10, 20, 30, 40, 50], "refund": [20, 30, 40, 50, 70]},
    "SPECIAL": {"change": None, "refund": [50, 60, 70, 80, 100]},
    "GROUP": {"change": None, "refund": None},
}
AIRLINE_OFFSET = {"MU": 0, "CA": 5, "CZ": 0, "HU": 5}
ROUTES = [("HGH", "PEK", "杭州", "北京", 1500), ("PEK", "HGH", "北京", "杭州", 1500),
          ("SHA", "CAN", "上海", "广州", 1300), ("PEK", "CTU", "北京", "成都", 1600),
          ("CAN", "HAK", "广州", "海口", 900)]
DAY_MULT = [1.0, 1.08, 0.95, 1.15, 1.0, 1.2, 0.9]
TAX = 50


def fee_table():
    out = {}
    for al, off in AIRLINE_OFFSET.items():
        for prod, rates in BASE_RATES.items():
            out[(al, prod)] = {k: (None if v is None else [min(100, p + off) if p > 0 else 0 for p in v])
                               for k, v in rates.items()}
    return out


def flight_no(ai: int, ri: int, al: str) -> str:
    return f"{al}{5101 + ri * 10 + ai}"


def fare_for(base: int, prod: str, day, ai: int) -> int:
    mult = DAY_MULT[(day.toordinal() + ai) % 7]
    return int(round(base * PRODUCT_FACTOR[prod] * mult / 10) * 10)


def flights(today):
    rows = []
    for ai, al in enumerate(AIRLINES):
        for ri, (o, d, oc, dc, base) in enumerate(ROUTES):
            dep_h = 7 + (ri * 3 + ai * 2) % 14
            for off in range(-1, 15):
                day = today + timedelta(days=off)
                rows.append({"flight_no": flight_no(ai, ri, al), "date": day.isoformat(), "airline": al,
                             "origin": o, "dest": d, "origin_city": oc, "dest_city": dc,
                             "dep_time": f"{dep_h:02d}:{ai * 15:02d}", "duration_min": 130,
                             "status": "scheduled", "delay_min": 0,
                             "fares": {p: fare_for(base, p, day, ai) for p in PRODUCTS}})
    return rows


# (user, airline, route_idx, product, day_offset, passenger, flight_status, delay)
ORDER_SPECS = [
    ("u_zhang", "MU", 0, "Y_DISCOUNT", 2, "张三", "scheduled", 0),
    ("u_zhang", "MU", 1, "Y_DISCOUNT", 5, "张三", "scheduled", 0),
    ("u_zhang", "CA", 2, "C", 10, "张三", "scheduled", 0),
    ("u_li", "CZ", 4, "SPECIAL", 1, "李四", "scheduled", 0),
    ("u_li", "HU", 3, "Y", 0, "李四", "scheduled", 0),        # 今天，起飞时间改为 now+3h
    ("u_li", "CA", 3, "F", 8, "李四", "scheduled", 0),
    ("u_wang", "MU", 2, "Y", 1, "王五", "cancelled", 0),
    ("u_wang", "CZ", 0, "Y_DISCOUNT", 3, "王五", "delayed", 150),
    ("u_wang", "HU", 4, "GROUP", 4, "王五", "scheduled", 0),
    ("u_zhang", "CZ", 2, "Y", -1, "张三", "scheduled", 0),    # 昨天，已误机
    ("u_li", "MU", 0, "Y", 6, "李四", "scheduled", 0),
    ("u_wang", "CA", 0, "SPECIAL", 2, "王五", "scheduled", 0),
]

PROFILES = [
    {"user_id": "u_zhang", "name": "张三", "tier": "金卡",
     "data": {"routes": {"HGH-PEK": 8, "PEK-HGH": 6}, "airlines": {"MU": 10}, "products": {"Y_DISCOUNT": 9},
              "prefer_action": {"CHANGE": 5, "REFUND": 1}, "price_sensitivity": "高"}},
    {"user_id": "u_li", "name": "李四", "tier": "银卡",
     "data": {"routes": {"PEK-CTU": 4}, "airlines": {"HU": 3, "CA": 2}, "products": {"Y": 3, "F": 2},
              "prefer_action": {"REFUND": 2}, "price_sensitivity": "中"}},
    {"user_id": "u_wang", "name": "王五", "tier": "普卡",
     "data": {"routes": {}, "airlines": {}, "products": {}, "prefer_action": {}, "price_sensitivity": "低"}},
]


def orders(today, now: datetime, flight_rows):
    # 注意：这里会直接修改 flight_rows（今天订单的起飞时间、各订单航班的状态），这是有意为之。
    # data.init() 在调用本函数之后才把航班写入数据库。
    by_key = {(f["flight_no"], f["date"]): f for f in flight_rows}
    out = []
    for i, (uid, al, ri, prod, off, pax, status, delay) in enumerate(ORDER_SPECS, start=1):
        ai = list(AIRLINES).index(al)
        f = by_key[(flight_no(ai, ri, al), (today + timedelta(days=off)).isoformat())]
        if off == 0:
            dep = now + timedelta(hours=3)
            # now+3h 跨午夜时落在 23:59，此时演示订单离起飞不足 3 小时
            f["dep_time"] = f"{dep.hour:02d}:{dep.minute:02d}" if dep.date() == today else "23:59"
        f["status"], f["delay_min"] = status, delay
        out.append({"order_id": f"ORD{today:%Y%m%d}{i:03d}", "user_id": uid, "passenger": pax,
                    "flight_no": f["flight_no"], "date": f["date"], "airline": al,
                    "cabin": PRODUCTS[prod][0], "fare_product": prod, "fare": f["fares"][prod],
                    "tax": TAX, "status": "ISSUED", "ticket_no": f"781-{2400000000 + i}"})
    return out


def _pct_text(p, verb):
    return f"免收{verb}费" if p == 0 else f"收取票面价{p}%的{verb}费"


def rules():
    out = []
    for (al, prod), table in fee_table().items():
        name, (cabin, plabel) = AIRLINES[al], PRODUCTS[prod]
        if table["change"] is None:
            chg = "该票价产品不得自愿改签，如需变更行程请办理退票后重新购票。"
        else:
            chg = "；".join(f"{WINDOW_LABEL[w]}申请改签，{_pct_text(p, '改期')}"
                           for w, p in zip(WINDOWS, table["change"]))
            chg += "。改签至票面价更高的航班需补齐差价，改签至票面价更低的航班不退差额。"
        if table["refund"] is None:
            rfd = "该票价产品不得自愿退票。"
        else:
            rfd = "；".join(f"{WINDOW_LABEL[w]}申请退票，{_pct_text(p, '退票')}"
                           for w, p in zip(WINDOWS, table["refund"]))
            rfd += f"。机场建设费{TAX}元在退票时全额退还。"
        for suffix, scen, kind, text in (("CHG", "VOLUNTARY_CHANGE", "自愿改签", chg),
                                         ("RFD", "VOLUNTARY_REFUND", "自愿退票", rfd)):
            out.append({"id": f"{al}-{prod}-{suffix}", "airline": al, "cabin": cabin, "fare_product": prod,
                        "scenario": scen, "title": f"{name}{plabel}（{prod}）{kind}规则", "content": text})
    out += [
        {"id": "GEN-CANCEL", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "CANCEL",
         "title": "航班取消（非旅客原因）退改规则",
         "content": "因航空公司原因航班取消，旅客可免费改签至同航线其他航班，或办理全额退票，免收改期费和退票费，改签不收取差价。"},
        {"id": "GEN-DELAY", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "DELAY",
         "title": "航班延误退改规则",
         "content": "航班延误2小时(含)以上，旅客可免费改签或全额退票，免收改期费和退票费，改签不收取差价；延误不足2小时按自愿退改规则执行。"},
        {"id": "GEN-NOSHOW", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "NO_SHOW",
         "title": "误机处理规则",
         "content": "旅客未在航班起飞前办理退改的视为误机，按各票价产品「起飞后」档位收取改期费或退票费。"},
        {"id": "GEN-FAREDIFF", "airline": "*", "cabin": "*", "fare_product": "*", "scenario": "FARE_DIFF",
         "title": "改签差价计算规则",
         "content": "改签差价按改签后航班同一票价产品的票面价减去原票面价计算，差价为正需补齐，差价为负不退还；改期费与差价分别计收。"},
    ]
    return out
