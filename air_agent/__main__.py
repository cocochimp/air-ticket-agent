"""命令行体验：uv run python -m air_agent --user u_zhang --order ORD...001"""
import argparse
import json
import uuid
from . import data, graph
from . import settings as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="u_zhang")
    ap.add_argument("--order", default=None)
    ap.add_argument("--mode", default=None, choices=[None, "offline", "deepseek"])
    args = ap.parse_args()
    data.init()
    sid = f"cli-{uuid.uuid4().hex[:8]}"
    print("订单：")
    for o in data.list_orders(args.user):
        print(" ", o["order_id"], o["flight_no"], o["date"], o["fare_product"], o["status"])
    print(f"模式：{args.mode or S.get()['mode']}。输入问题，空行退出。")
    pending = None
    while True:
        q = input("\n你：").strip()
        if not q:
            break
        if pending and pending["kind"] == "clarify":
            res = graph.resume(sid, {"value": q})
        elif pending and pending["kind"] == "confirm" and q in ("确认", "y"):
            res = graph.resume(sid, {"confirm": True, "index": pending["options"][0]["index"]})
        elif pending and pending["kind"] == "confirm" and q in ("取消", "n"):
            res = graph.resume(sid, {"confirm": False})
        else:
            res = graph.run_turn(sid, args.user, q, order_id=args.order, mode=args.mode)
        for e in res["trace"]:
            print(f"  [{e['status']:>7}] {e['node']:<16} {e['note']}")
        pending = res["interrupt"]
        if res.get("final", {}).get("text"):
            print("助手：", res["final"]["text"])
        if pending:
            print("等待：", json.dumps({k: pending[k] for k in pending if k != "options"}, ensure_ascii=False))
            for opt in pending.get("options", []):
                print("   -", opt.get("label") or opt.get("title"), opt.get("value") or opt.get("total") or "")


if __name__ == "__main__":
    main()
