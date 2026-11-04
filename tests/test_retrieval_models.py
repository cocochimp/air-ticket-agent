import pytest
from air_agent import data, retrieval
from air_agent import settings as S

pytestmark = pytest.mark.slow


def test_hybrid_backend_with_real_models():
    S.update({"retrieval": {"backend": "hybrid"}})
    data.init()
    retrieval.invalidate()
    r = retrieval.search("打折票改到后天要交多少改期费", {"airline": "MU", "fare_product": "Y_DISCOUNT",
                                                    "scenario": "VOLUNTARY_CHANGE"})
    assert r["backend"] == "hybrid" and not r["degraded"]
    assert r["rules"][0]["id"] == "MU-Y_DISCOUNT-CHG"
