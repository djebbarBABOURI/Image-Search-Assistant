import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "chatbot"))

from scoring import call_matches, score_query, summarize  # noqa: E402

DEFAULTS = {"classify": {"top_k": 3}, "count_by": {"label": "", "min_confidence": 0.0}}
QUERIES = json.loads((ROOT / "tests" / "chatbot_queries.json").read_text(encoding="utf-8"))
TOOLS = {"classify", "find_similar", "filter_images", "count_by", "top_confusions", "list_breeds"}


def call(name, **args):
    return {"name": name, "args": args}


def test_query_set_is_well_formed():
    assert len(QUERIES) >= 20
    assert len({q["id"] for q in QUERIES}) == len(QUERIES)
    assert sum(len(q["expected"]) >= 2 for q in QUERIES) >= 5
    for q in QUERIES:
        assert q["query"].strip()
        for e in q["expected"]:
            assert e["tool"] in TOOLS


def test_args_are_normalized():
    exp = {"tool": "count_by", "args": {"label": "British_Shorthair", "min_confidence": 0.9}}
    assert call_matches(exp, call("count_by", label="british shorthair", min_confidence="0.9"))
    assert not call_matches(exp, call("count_by", label="Bengal", min_confidence=0.9))
    assert not call_matches(exp, call("filter_images", label="British_Shorthair"))


def test_missing_arg_accepted_only_if_default():
    assert call_matches({"tool": "classify", "args": {"top_k": 3}}, call("classify", image_id="x"), DEFAULTS)
    assert not call_matches({"tool": "classify", "args": {"top_k": 5}}, call("classify", image_id="x"), DEFAULTS)
    assert not call_matches({"tool": "classify", "args": {"image_id": "x"}}, call("classify"), DEFAULTS)


def test_image_extension_ignored():
    exp = {"tool": "classify", "args": {"image_id": "Abyssinian_1"}}
    assert call_matches(exp, call("classify", image_id="Abyssinian_1.jpg"))


def test_chain_order():
    q = {"expected": [{"tool": "classify", "args": {"image_id": "a"}}, {"tool": "count_by"}]}
    good = score_query(q, [call("classify", image_id="a"), call("count_by", label="boxer")])
    assert good["calls_ok"] and good["extra"] == 0
    reversed_ = score_query(q, [call("count_by", label="boxer"), call("classify", image_id="a")])
    assert reversed_["tools_ok"] and not reversed_["calls_ok"] and reversed_["matched"] == 1

    unordered = score_query({**q, "ordered": False},
                            [call("count_by", label="boxer"), call("classify", image_id="a")])
    assert unordered["calls_ok"]


def test_no_tool_expected():
    q = {"expected": []}
    assert score_query(q, [])["calls_ok"]
    bad = score_query(q, [call("list_breeds")])
    assert not bad["calls_ok"] and bad["extra"] == 1


def test_wrong_args_and_extra_calls():
    q = {"expected": [{"tool": "find_similar", "args": {"image_id": "pug_12", "k": 5}}]}
    s = score_query(q, [call("list_breeds"), call("find_similar", image_id="pug_12", k=3)])
    assert s["tools_ok"] and not s["calls_ok"] and s["extra"] == 2


def test_summarize():
    scores = [
        {"tools_ok": True, "calls_ok": True, "matched": 2, "n_expected": 2, "extra": 0},
        {"tools_ok": True, "calls_ok": False, "matched": 0, "n_expected": 1, "extra": 1},
        {"tools_ok": False, "calls_ok": False, "matched": 0, "n_expected": 0, "extra": 1},
        {"tools_ok": True, "calls_ok": True, "matched": 1, "n_expected": 1, "extra": 0},
    ]
    s = summarize(scores)
    assert s["tool_selection_rate"] == 0.75
    assert s["correct_call_rate"] == 0.5
    assert s["expected_call_recall"] == 0.75
    assert s["chain_success_rate"] == 1.0 and s["n_chains"] == 1
    assert s["extra_calls"] == 2
