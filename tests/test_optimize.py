import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_logic.optimize import EvalResult, _dominates, _update_frontier


def test_eval_result_defaults():
    r = EvalResult(artifact="foo", score=0.5)
    assert r.secondary_score == 0.0
    assert r.asi == ""
    assert r.asi_kind == "proof"


def test_dominates_strictly_better():
    a = EvalResult(artifact="a", score=0.9, secondary_score=0.8)
    b = EvalResult(artifact="b", score=0.7, secondary_score=0.7)
    assert _dominates(a, b)
    assert not _dominates(b, a)


def test_dominates_equal_is_not_domination():
    a = EvalResult(artifact="a", score=0.9, secondary_score=0.8)
    assert not _dominates(a, a)


def test_dominates_tradeoff_is_not_dominated():
    a = EvalResult(artifact="a", score=0.9, secondary_score=0.6)  # high F1, low prec
    b = EvalResult(artifact="b", score=0.7, secondary_score=0.9)  # low F1, high prec
    assert not _dominates(a, b)
    assert not _dominates(b, a)


def test_update_frontier_adds_non_dominated():
    f = _update_frontier([], EvalResult(artifact="x", score=0.8, secondary_score=0.7), frontier_size=5)
    assert len(f) == 1


def test_update_frontier_prunes_dominated():
    r1 = EvalResult(artifact="x", score=0.5, secondary_score=0.5)
    r2 = EvalResult(artifact="y", score=0.9, secondary_score=0.9)
    f = _update_frontier([r1], r2, frontier_size=5)
    assert len(f) == 1
    assert f[0].artifact == "y"


def test_update_frontier_keeps_tradeoff_pair():
    r1 = EvalResult(artifact="high_f1", score=0.9, secondary_score=0.6)
    r2 = EvalResult(artifact="high_prec", score=0.7, secondary_score=0.9)
    f = _update_frontier([r1], r2, frontier_size=5)
    assert len(f) == 2


def test_update_frontier_respects_size_cap():
    f = []
    for i in range(10):
        # non-dominated tradeoff points along the Pareto front
        f = _update_frontier(
            f,
            EvalResult(artifact=str(i), score=i * 0.1, secondary_score=1.0 - i * 0.1),
            frontier_size=5,
        )
    assert len(f) <= 5


def test_update_frontier_tiebreak_shorter_artifact():
    r1 = EvalResult(artifact="long_name", score=0.8, secondary_score=0.8)
    r2 = EvalResult(artifact="short", score=0.8, secondary_score=0.8)
    # r2 is not dominated by r1 (equal scores), so both enter; shorter wins sort position
    f = _update_frontier([r1], r2, frontier_size=5)
    assert f[0].artifact == "short"
