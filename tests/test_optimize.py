import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor_logic.optimize import EvalResult, _dominates, _update_frontier, optimize


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


def test_optimize_accept_fires_first_step():
    def propose(fb): return "perfect"
    def evaluate(art): return EvalResult(artifact=art, score=1.0)
    def accept(r): return r.score >= 1.0
    frontier = optimize(propose, evaluate, accept, pareto_axes=("f1", "prec"))
    assert len(frontier) == 1
    assert frontier[0].score == 1.0


def test_optimize_max_steps_respected():
    calls = [0]
    def propose(fb): calls[0] += 1; return f"art{calls[0]}"
    def evaluate(art): return EvalResult(artifact=art, score=0.5)
    def accept(r): return False
    optimize(propose, evaluate, accept, pareto_axes=("f1", "prec"), max_steps=7, stagnation_k=100)
    assert calls[0] == 7


def test_optimize_stagnation_exit():
    calls = [0]
    def propose(fb): calls[0] += 1; return "same_artifact"
    def evaluate(art): return EvalResult(artifact=art, score=0.5)
    def accept(r): return False
    optimize(propose, evaluate, accept, pareto_axes=("f1", "prec"),
             max_steps=50, stagnation_k=3)
    # Step 0: prev={}, curr={"same"} → different; steps 1,2,3: stagnation count 1,2,3 → exit
    assert calls[0] == 4


def test_optimize_feedback_is_last_asi():
    received_feedback = []
    def propose(fb):
        received_feedback.append(fb)
        return f"art{len(received_feedback)}"
    def evaluate(art):
        return EvalResult(artifact=art, score=0.3, asi=f"why_{art}", asi_kind="why_not")
    def accept(r): return len(received_feedback) >= 3
    optimize(propose, evaluate, accept, pareto_axes=("f1", "prec"))
    # First call gets empty string; subsequent calls get ASI from last evaluation
    assert received_feedback[0] == ""
    assert received_feedback[1] == "why_art1"
    assert received_feedback[2] == "why_art2"


def test_optimize_returns_pareto_frontier_not_just_best():
    """Non-dominated tradeoff candidates both survive."""
    artifacts = iter([
        ("high_f1",   0.9, 0.6),
        ("high_prec", 0.7, 0.9),
        ("dominated", 0.5, 0.5),
    ])
    def propose(fb):
        try: return next(artifacts)[0]
        except StopIteration: return "same"
    scores = {"high_f1": (0.9, 0.6), "high_prec": (0.7, 0.9), "dominated": (0.5, 0.5), "same": (0.5, 0.5)}
    def evaluate(art):
        s = scores[art]
        return EvalResult(artifact=art, score=s[0], secondary_score=s[1])
    def accept(r): return False
    frontier = optimize(propose, evaluate, accept, pareto_axes=("f1", "prec"),
                        max_steps=10, frontier_size=5, stagnation_k=4)
    arts = {r.artifact for r in frontier}
    assert "high_f1" in arts
    assert "high_prec" in arts
    assert "dominated" not in arts
