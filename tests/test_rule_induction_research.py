from experiments import exp79_self_play_loop as exp79

from tensor_logic.research.constants import CONTACTS, QUERY_TARGETS
from tensor_logic.research.rule_induction import (
    Config,
    Guide,
    RuleInductionLoop,
    RuleKB,
    run_self_play,
)


def tiny_config(**overrides):
    values = dict(
        name="test",
        schema=CONTACTS,
        n_pos=6,
        n_neg=6,
        noise=0.0,
        n_entities=24,
        max_steps=1,
        min_equiv=0.95,
        f1_threshold=0.80,
        n_attempts=1,
        vote_threshold=0.70,
        min_support=1,
        exclusive=False,
    )
    values.update(overrides)
    return Config(**values)


def stable_projection(result):
    return {
        "mode": result["mode"],
        "answered_at_0": result["answered_at_0"],
        "answered_after": result["answered_after"],
        "total_queries": result["total_queries"],
        "rules_induced": result["rules_induced"],
        "steps_taken": result["steps_taken"],
        "rules": result["rules"],
        "step_log": [
            {k: v for k, v in entry.items() if k != "elapsed_s"}
            for entry in result["step_log"]
        ],
    }


def test_exp79_uses_reusable_research_module():
    assert exp79.run_self_play is run_self_play
    assert exp79.KB is RuleKB
    assert exp79.RuleInductionLoop is RuleInductionLoop


def test_negative_proof_triggers_induction_before_rules_exist():
    loop = RuleInductionLoop(tiny_config(), seed=42)
    target_name, _ = QUERY_TARGETS[0]
    assert loop.kb.has_rule(target_name) is False
    assert loop.solver.needs_induction(target_name) is True


def test_conjecturer_is_deterministic_for_same_world_and_step():
    left = RuleInductionLoop(tiny_config(), seed=42)
    right = RuleInductionLoop(tiny_config(), seed=42)
    target_name, _ = QUERY_TARGETS[0]

    proposal_left = left.conjecturer.propose(left.gold[target_name], step=0)
    proposal_right = right.conjecturer.propose(right.gold[target_name], step=0)

    assert proposal_left == proposal_right


def test_guide_rejects_semantically_wrong_rule():
    loop = RuleInductionLoop(tiny_config(), seed=42)
    target_name, gold_body = QUERY_TARGETS[0]
    guide = Guide(loop.cfg, loop.kb, heldout_seeds=[100])

    result = guide.evaluate_candidate(
        target_name,
        ["knows"],
        gold_body,
        f1_score=1.0,
    )

    assert result["accepted"] is False
    assert result["outcome"] == "failed_equiv"
    assert loop.kb.has_rule(target_name) is False


def test_rule_kb_compiles_transpose_into_canonical_tl_syntax():
    assert (
        RuleKB.to_tl_rule(
            "managed_peer",
            ["manages^T", "manages"],
        )
        == "managed_peer(X, Z) := manages(Y, X) * manages(Y, Z)"
    )


def test_self_play_reproducible_ignoring_elapsed_time():
    cfg = tiny_config()
    first = run_self_play(cfg, seed=7)
    second = run_self_play(cfg, seed=7)

    assert stable_projection(first) == stable_projection(second)
    assert first["steps_taken"] <= cfg.max_steps
