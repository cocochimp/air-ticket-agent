from air_agent import data, memory
from air_agent import settings as S

D = "20260926"


def setup_function(_):
    data.init()


def test_order_stack_dedup_and_cap():
    mem = memory.load_session("s1")
    for i in [1, 2, 3, 1, 4, 5, 6]:
        memory.push_order(mem, f"ORD{D}{i:03d}")
    assert mem["order_stack"][0] == f"ORD{D}006"
    assert len(mem["order_stack"]) == 5 and mem["order_stack"].count(f"ORD{D}001") == 1


def test_sliding_window_summary():
    mem = memory.load_session("s2")
    for i in range(6):
        memory.add_turn(mem, "user", f"第{i}轮：改到大后天")
        memory.add_turn(mem, "assistant", f"第{i}轮答复：改期费 100 元")
    assert len(mem["turns"]) == 8
    assert "第0轮" in mem["summary"] and "第5轮" not in mem["summary"]


def test_session_roundtrip():
    mem = memory.load_session("s3")
    memory.push_order(mem, f"ORD{D}001")
    mem["last_intent"] = "CHANGE"
    memory.save_session("s3", "u_zhang", mem)
    again = memory.load_session("s3")
    assert again["order_stack"] == [f"ORD{D}001"] and again["last_intent"] == "CHANGE"


def test_profile_update_is_deduplicated():
    o = data.get_order(f"ORD{D}003")
    before = data.get_profile("u_zhang")["data"]["airlines"].get("CA", 0)
    memory.update_profile("u_zhang", o, "REFUND", key="s4:t1")
    memory.update_profile("u_zhang", o, "REFUND", key="s4:t1")
    prof = data.get_profile("u_zhang")["data"]
    assert prof["airlines"]["CA"] == before + 1
    assert prof["prefer_action"]["REFUND"] == 2


def test_profile_boost():
    assert memory.profile_boost("u_zhang") == {"airlines": ["MU"], "products": ["Y_DISCOUNT"]}
    assert memory.profile_boost("u_wang") == {"airlines": [], "products": []}


def test_clear_long_memory():
    memory.clear_profile("u_zhang")
    assert data.get_profile("u_zhang")["data"]["airlines"] == {}


def test_token_budget():
    S.update({"memory": {"token_budget": 600}})
    mem = memory.load_session("s5")
    for i in range(10):
        memory.add_turn(mem, "user", "我想把杭州飞北京那张票改到大后天，顺便问下退票多少钱" * 2)
    rules = [{"id": f"R{i}", "title": "规则", "content": "起飞前7天以上申请改签收取票面价5%的改期费。" * 6}
             for i in range(10)]
    ctx = memory.build_context(data.get_order(f"ORD{D}001"), rules, mem, data.get_profile("u_zhang"))
    assert ctx["total"] <= 600
    assert ctx["full_total"] > ctx["total"] and ctx["saved_pct"] > 0
    assert ctx["blocks"]["order"]["truncated"] is False
    assert set(ctx["blocks"]) == {"system", "order", "rules", "history", "profile"}
