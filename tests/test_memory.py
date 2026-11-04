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


# ---------------------------------------------------------------- 加固
import threading
import pytest


def _rules(n=10):
    return [{"id": f"R{i}", "title": "规则", "content": "起飞前7天以上申请改签收取票面价5%的改期费。" * 6}
            for i in range(n)]


def test_history_keeps_newest_turns_after_many_turns():
    S.update({"memory": {"token_budget": 600}})
    mem = memory.load_session("h1")
    for i in range(30):
        memory.add_turn(mem, "user", f"第{i}轮用户提问：我想把杭州飞北京那张票改到大后天并问问退票费")
        memory.add_turn(mem, "assistant", f"第{i}轮答复：改期费按规则收取")
    memory.add_turn(mem, "user", "最后一句：帮我退掉这张票")
    ctx = memory.build_context(data.get_order(f"ORD{D}001"), _rules(), mem, data.get_profile("u_zhang"))
    assert "最后一句：帮我退掉这张票" in ctx["blocks"]["history"]["text"]
    assert ctx["total"] <= 600
    assert len(mem["summary"]) <= 600 and len(mem["archive"]) <= 50


def test_history_uses_leftover_for_summary_tail():
    S.update({"memory": {"token_budget": 3000}})
    mem = memory.load_session("h2")
    for i in range(12):
        memory.add_turn(mem, "user", f"第{i}轮：改签")
    ctx = memory.build_context(data.get_order(f"ORD{D}001"), [], mem, None)
    text = ctx["blocks"]["history"]["text"]
    assert "早前摘要" in text and "第11轮：改签" in text and not ctx["blocks"]["history"]["truncated"]


def test_carry_over_gives_unused_share_to_later_blocks():
    S.update({"memory": {"token_budget": 1000}})
    mem = memory.load_session("h3")
    for i in range(8):
        memory.add_turn(mem, "user", "我想把杭州飞北京那张票改到大后天，顺便问下退票多少钱" * 3)
    order, prof = data.get_order(f"ORD{D}001"), data.get_profile("u_zhang")
    with_rules = memory.build_context(order, _rules(), mem, prof)
    no_rules = memory.build_context(order, [], mem, prof)
    base = int(1000 * S.get()["memory"]["budget_share"]["history"])
    assert no_rules["blocks"]["history"]["budget"] > with_rules["blocks"]["history"]["budget"]
    assert no_rules["blocks"]["history"]["budget"] > base
    for ctx in (with_rules, no_rules):
        assert ctx["total"] <= 1000
        assert all(b["tokens"] <= b["budget"] for b in ctx["blocks"].values())


def test_system_and_order_never_truncated_at_minimum_budget():
    S.update({"memory": {"token_budget": 200}})
    mem = memory.load_session("h4")
    for i in range(10):
        memory.add_turn(mem, "user", "我想把杭州飞北京那张票改到大后天" * 3)
    ctx = memory.build_context(data.get_order(f"ORD{D}001"), _rules(), mem, data.get_profile("u_zhang"))
    assert ctx["blocks"]["system"]["truncated"] is False and ctx["blocks"]["order"]["truncated"] is False
    assert ctx["total"] <= 200


def test_settings_reject_tiny_token_budget():
    before = S.get()["memory"]["token_budget"]
    with pytest.raises(ValueError):
        S.update({"memory": {"token_budget": 199}})
    assert S.get()["memory"]["token_budget"] == before
    S.update({"memory": {"token_budget": 200}})
    assert S.get()["memory"]["token_budget"] == 200


def test_profile_update_is_atomic_under_threads():
    o = data.get_order(f"ORD{D}003")
    before = data.get_profile("u_zhang")["data"]["airlines"].get("CA", 0)
    errors = []

    def work(i):
        try:
            memory.update_profile("u_zhang", o, "REFUND", key=f"thread:{i}")
        except Exception as e:      # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    assert data.get_profile("u_zhang")["data"]["airlines"]["CA"] == before + 20


def test_session_lock_is_per_session():
    assert memory.session_lock("a") is memory.session_lock("a")
    assert memory.session_lock("a") is not memory.session_lock("b")


def test_internal_keys_hidden_from_public_profile_and_context():
    o = data.get_order(f"ORD{D}003")
    memory.update_profile("u_zhang", o, "REFUND", key="k1")
    prof = data.get_profile("u_zhang")
    assert "_seen" in prof["data"]
    pub = memory.public_profile(prof)
    assert "_seen" not in pub["data"] and pub["name"] == prof["name"] and "_seen" in prof["data"]
    mem = memory.load_session("h5")
    a = memory.build_context(o, [], mem, prof)["full_total"]
    prof["data"]["_seen"] = [f"key-{i}" for i in range(200)]
    assert memory.build_context(o, [], mem, prof)["full_total"] == a


def test_profile_text_uses_labels_not_codes():
    text = memory.profile_text(data.get_profile("u_zhang"))
    assert "东方云航" in text and "经济舱折扣" in text and "改签" in text
    assert "MU" not in text and "Y_DISCOUNT" not in text and "CHANGE" not in text
