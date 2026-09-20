"""Reusable conjecturer / solver / guide loop for Tensor Logic rule induction.

This module extracts the exp79 self-play machinery from an experiment script so
new research can reuse the same acceptance semantics without copying plumbing.

The roles are deliberately separated:
- Conjecturer proposes candidate rule bodies from examples.
- Solver decides which unresolved target currently needs a rule.
- Guide admits only candidates that pass semantic-equivalence, stability, and
  held-out generalization checks.

Candidate rules are hypotheses. RuleKB is the only admission boundary that
turns a candidate into executable Tensor Logic state.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import torch

from tensor_logic import Program, prove_negative
from tensor_logic.research.constants import CONTACTS, QUERY_TARGETS
from tensor_logic.research.utils import (
    Schema,
    apply_body,
    corrupt_examples,
    enumerate_rules,
    f1 as compute_f1,
    fixpoint_stable,
    gen_world,
    induce_from_examples,
    sample_examples,
    sample_exclusive_examples,
    semantic_equiv,
)


@dataclass(frozen=True)
class Config:
    """Configuration for one self-play rule-induction run."""

    name: str
    schema: Schema
    n_pos: int
    n_neg: int
    noise: float
    n_entities: int
    max_steps: int
    min_equiv: float
    f1_threshold: float
    n_attempts: int
    vote_threshold: float
    min_support: int
    exclusive: bool = False


def candidate_relation_names(schema: Schema) -> list[str]:
    """Return composable binary person-person relations."""
    return [
        name
        for name, types in schema.relations.items()
        if types == ("person", "person")
    ]


def candidate_prediction_cache(
    base: dict[str, torch.Tensor],
    rel_names: list[str],
    max_len: int = 3,
) -> tuple[list[tuple[str, ...]], torch.Tensor]:
    """Precompute each candidate body's tensor for fast repeated scoring."""
    keys: list[tuple[str, ...]] = []
    preds: list[torch.Tensor] = []
    for body in enumerate_rules(rel_names, max_len=max_len):
        try:
            preds.append(apply_body(body, base))
            keys.append(tuple(body))
        except KeyError:
            continue
    if not preds:
        raise ValueError("candidate rule search space is empty")
    return keys, torch.stack(preds)


def cross_world_generalization(
    body: list[str],
    gold_body: list[str],
    schema: Schema,
    seeds: list[int],
) -> dict[int, int]:
    """Check exact semantic agreement on independently generated worlds."""
    results: dict[int, int] = {}
    for seed in seeds:
        base = gen_world(schema, n_entities=15, seed=seed)
        base_clean = {
            key: value
            for key, value in base.items()
            if key in schema.relations or key.rstrip("^T") in schema.relations
        }
        try:
            pred = apply_body(body, base_clean)
            gold = apply_body(gold_body, base_clean)
            results[seed] = int(torch.equal(pred, gold))
        except KeyError:
            results[seed] = 0
    return results


class RuleKB:
    """Executable rule store backed by the canonical Tensor Logic Program."""

    def __init__(self, base: dict[str, torch.Tensor], schema: Schema, n: int):
        self.base = base
        self.schema = schema
        self.n = n
        self.rules: list[tuple[str, list[str]]] = []

        self.program = Program()
        self.symbols = [f"p{i}" for i in range(n)]
        self.domain = self.program.domain("person", self.symbols)

        for name, tensor in base.items():
            if name not in self.program.relations:
                self.program.relation(name, "person", "person")
            for row, col in tensor.nonzero():
                self.program.fact(name, f"p{row.item()}", f"p{col.item()}")

        for target_name, _ in QUERY_TARGETS:
            if target_name not in self.program.relations:
                self.program.relation(target_name, "person", "person")

    def derive(self, name: str) -> torch.Tensor:
        for rule_name, body in self.rules:
            if rule_name == name:
                return apply_body(body, self.base)
        return torch.zeros(self.n, self.n)

    def has_rule(self, name: str) -> bool:
        return any(rule_name == name for rule_name, _ in self.rules)

    @staticmethod
    def to_tl_rule(name: str, body: list[str]) -> str:
        """Compile a relation-chain body into the canonical TL rule syntax."""
        if not body:
            raise ValueError("rule body must contain at least one relation")
        vars_ = ["X", "Y", "Z", "W", "V", "U", "T"]
        if len(body) >= len(vars_):
            raise ValueError("rule body is too long for the chain compiler")

        body_parts: list[str] = []
        for idx, rel in enumerate(body):
            src_var = vars_[idx]
            dst_var = vars_[idx + 1]
            if rel.endswith("^T"):
                body_parts.append(f"{rel[:-2]}({dst_var}, {src_var})")
            else:
                body_parts.append(f"{rel}({src_var}, {dst_var})")
        return f"{name}(X, {vars_[len(body)]}) := " + " * ".join(body_parts)

    def add_rule(self, name: str, body: list[str]) -> bool:
        """Admit a candidate only if adding it reaches a stable fixpoint."""
        candidate = (name, list(body))
        if not fixpoint_stable(self.base, self.rules + [candidate]):
            return False

        self.program.rule(self.to_tl_rule(name, body))
        self.rules.append(candidate)
        return True


class Conjecturer:
    """Generate candidate rules from noisy positive and negative examples."""

    def __init__(
        self,
        cfg: Config,
        base: dict[str, torch.Tensor],
        rel_names: list[str],
        *,
        max_len: int = 3,
    ):
        self.cfg = cfg
        self.base = base
        self.rel_names = rel_names
        self.body_keys, self.pred_stack = candidate_prediction_cache(
            base, rel_names, max_len=max_len
        )
        self.body_index = {body: i for i, body in enumerate(self.body_keys)}

    def propose_from_examples(
        self,
        positive: list[tuple[int, int]],
        negative: list[tuple[int, int]],
        *,
        target: Optional[torch.Tensor] = None,
    ) -> dict:
        """Rank candidates against caller-supplied examples.

        This is the reusable path for research settings where valid examples
        occupy a constrained subspace (for example, block-diagonal worlds).
        The caller chooses examples; the Conjecturer still owns candidate
        enumeration, scoring, and deterministic tie-breaking.
        """
        if not positive:
            return {
                "body": None,
                "f1": 0.0,
                "example_f1": 0.0,
                "scores": {},
            }

        pos_i = torch.tensor(
            [pair[0] for pair in positive],
            dtype=torch.long,
        )
        pos_j = torch.tensor(
            [pair[1] for pair in positive],
            dtype=torch.long,
        )
        neg_i = torch.tensor(
            [pair[0] for pair in negative],
            dtype=torch.long,
        )
        neg_j = torch.tensor(
            [pair[1] for pair in negative],
            dtype=torch.long,
        )

        tp = self.pred_stack[:, pos_i, pos_j].sum(dim=1)
        if negative:
            fp = self.pred_stack[:, neg_i, neg_j].sum(dim=1)
        else:
            fp = torch.zeros_like(tp)
        fn = len(positive) - tp
        precision = tp / (tp + fp).clamp_min(1e-9)
        recall = tp / (tp + fn).clamp_min(1e-9)
        scores = (
            2
            * precision
            * recall
            / (precision + recall).clamp_min(1e-9)
        )

        best_score = float(scores.max().item())
        best_indices = (
            scores == scores.max()
        ).nonzero().flatten().tolist()
        best_idx = min(
            best_indices,
            key=lambda idx: (
                len(
                    {
                        rel.rstrip("^T")
                        for rel in self.body_keys[idx]
                    }
                ),
                len(self.body_keys[idx]),
                self.body_keys[idx],
            ),
        )
        best_key = self.body_keys[best_idx]
        prediction = self.pred_stack[best_idx]

        return {
            "body": list(best_key),
            "f1": (
                compute_f1(prediction, target)
                if target is not None
                else best_score
            ),
            "example_f1": best_score,
            "scores": {
                "|".join(body): float(scores[idx].item())
                for idx, body in enumerate(self.body_keys)
            },
        }

    def propose(self, target: torch.Tensor, *, step: int) -> dict:
        """Return the majority-vote candidate for one target tensor."""
        votes: dict[tuple[str, ...], int] = {}

        for attempt in range(self.cfg.n_attempts):
            attempt_seed = step * 100 + attempt
            if self.cfg.exclusive:
                pos, neg = sample_exclusive_examples(
                    target,
                    self.base,
                    self.cfg.n_pos,
                    self.cfg.n_neg,
                    attempt_seed,
                )
            else:
                pos, neg = sample_examples(
                    target,
                    self.cfg.n_pos,
                    self.cfg.n_neg,
                    attempt_seed,
                )

            if not pos:
                continue

            pos, neg = corrupt_examples(
                pos, neg, self.cfg.noise, seed=attempt_seed
            )
            pos_i = torch.tensor([ij[0] for ij in pos], dtype=torch.long)
            pos_j = torch.tensor([ij[1] for ij in pos], dtype=torch.long)
            neg_i = torch.tensor([ij[0] for ij in neg], dtype=torch.long)
            neg_j = torch.tensor([ij[1] for ij in neg], dtype=torch.long)

            tp = self.pred_stack[:, pos_i, pos_j].sum(dim=1)
            fp = self.pred_stack[:, neg_i, neg_j].sum(dim=1)
            fn = len(pos) - tp
            precision = tp / (tp + fp).clamp_min(1e-9)
            recall = tp / (tp + fn).clamp_min(1e-9)
            scores = (
                2
                * precision
                * recall
                / (precision + recall).clamp_min(1e-9)
            )

            for idx in (
                scores >= self.cfg.vote_threshold
            ).nonzero().flatten().tolist():
                body_key = self.body_keys[idx]
                votes[body_key] = votes.get(body_key, 0) + 1

        qualified = {
            body: count
            for body, count in votes.items()
            if count >= self.cfg.min_support
        }
        if not qualified:
            return {"body": None, "f1": 0.0, "votes": votes}

        max_votes = max(qualified.values())
        best_key = min(
            (body for body, count in qualified.items() if count == max_votes),
            key=lambda body: (
                len({rel.rstrip("^T") for rel in body}),
                len(body),
                body,
            ),
        )
        best_body = list(best_key)
        pred = self.pred_stack[self.body_index[best_key]]
        return {
            "body": best_body,
            "f1": compute_f1(pred, target),
            "votes": votes,
        }


class Solver:
    """Choose unresolved targets and determine whether accepted rules answer them."""

    def __init__(
        self,
        cfg: Config,
        kb: RuleKB,
        gold: dict[str, torch.Tensor],
    ):
        self.cfg = cfg
        self.kb = kb
        self.gold = gold

    def needs_induction(self, name: str) -> bool:
        target = self.gold[name]
        pos_indices = target.nonzero()

        if len(pos_indices) == 0:
            neg_proof = prove_negative(self.kb.program, name, "p0", "p1")
        else:
            idx = pos_indices[0]
            src = f"p{idx[0].item()}"
            dst = f"p{idx[1].item()}"
            neg_proof = prove_negative(self.kb.program, name, src, dst)

        return (
            neg_proof is not None
            and neg_proof.reason == "no_rules"
        )

    def answerable(self, name: str) -> bool:
        if not self.kb.has_rule(name):
            return False
        pred = self.kb.derive(name)
        return compute_f1(pred, self.gold[name]) >= self.cfg.f1_threshold

    def next_target(self) -> Optional[tuple[str, list[str]]]:
        return next(
            (
                (name, body)
                for name, body in QUERY_TARGETS
                if self.needs_induction(name)
            ),
            None,
        )


class Guide:
    """Zero-trust admission gate for candidate rules."""

    def __init__(
        self,
        cfg: Config,
        kb: RuleKB,
        *,
        heldout_schema: Schema = CONTACTS,
        heldout_seeds: Optional[list[int]] = None,
    ):
        self.cfg = cfg
        self.kb = kb
        self.heldout_schema = heldout_schema
        self.heldout_seeds = (
            heldout_seeds
            if heldout_seeds is not None
            else [100, 101, 102, 103, 104]
        )

    def evaluate_candidate(
        self,
        target_name: str,
        candidate_body: Optional[list[str]],
        gold_body: list[str],
        *,
        f1_score: float,
    ) -> dict:
        """Evaluate, admit, and independently generalize one candidate."""
        result = {
            "outcome": "failed_f1",
            "accepted": False,
            "equiv": 0.0,
            "gen_scores": None,
        }
        if candidate_body is None:
            return result

        equiv = semantic_equiv(
            candidate_body,
            gold_body,
            self.cfg.schema,
            n_worlds=30,
            n_entities=12,
        )
        result["equiv"] = equiv
        if equiv < self.cfg.min_equiv:
            result["outcome"] = "failed_equiv"
            return result

        accepted = self.kb.add_rule(target_name, candidate_body)
        result["accepted"] = accepted
        if not accepted:
            result["outcome"] = "rejected_unstable"
            return result

        result["outcome"] = "accepted"
        result["gen_scores"] = cross_world_generalization(
            candidate_body,
            gold_body,
            self.heldout_schema,
            self.heldout_seeds,
        )
        return result


class RuleInductionLoop:
    """Reusable conjecturer -> solver -> guide research loop."""

    def __init__(self, cfg: Config, *, seed: int = 42):
        self.cfg = cfg
        self.seed = seed
        self.base = gen_world(
            cfg.schema,
            n_entities=cfg.n_entities,
            density=0.08,
            seed=seed,
        )
        self.contacts_base = {
            key: value
            for key, value in self.base.items()
            if key in CONTACTS.relations
        }
        self.gold = {
            name: apply_body(body, self.contacts_base)
            for name, body in QUERY_TARGETS
        }
        self.kb = RuleKB(self.base, cfg.schema, cfg.n_entities)
        self.rel_names = candidate_relation_names(cfg.schema)
        self.conjecturer = Conjecturer(cfg, self.base, self.rel_names)
        self.solver = Solver(cfg, self.kb, self.gold)
        self.guide = Guide(cfg, self.kb)

    def run(self) -> dict:
        answered_at_0 = sum(
            1
            for name, _ in QUERY_TARGETS
            if self.solver.answerable(name)
        )
        step_log: list[dict] = []

        for step in range(self.cfg.max_steps):
            target = self.solver.next_target()
            if target is None:
                break

            target_name, gold_body = target
            started = time.perf_counter()
            proposal = self.conjecturer.propose(
                self.gold[target_name],
                step=step,
            )
            judgment = self.guide.evaluate_candidate(
                target_name,
                proposal["body"],
                gold_body,
                f1_score=proposal["f1"],
            )

            step_log.append(
                {
                    "step": step,
                    "target": target_name,
                    "outcome": judgment["outcome"],
                    "induced": proposal["body"],
                    "gold": gold_body,
                    "f1": round(proposal["f1"], 3),
                    "equiv": round(judgment["equiv"], 3),
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "gen_scores": judgment["gen_scores"],
                }
            )

        answered_after = sum(
            1
            for name, _ in QUERY_TARGETS
            if self.solver.answerable(name)
        )
        return {
            "mode": self.cfg.name,
            "answered_at_0": answered_at_0,
            "answered_after": answered_after,
            "total_queries": len(QUERY_TARGETS),
            "rules_induced": len(self.kb.rules),
            "steps_taken": len(step_log),
            "step_log": step_log,
            "rules": [(name, body) for name, body in self.kb.rules],
        }


def run_self_play(cfg: Config, seed: int = 42) -> dict:
    """Compatibility wrapper used by exp79 and future experiments."""
    return RuleInductionLoop(cfg, seed=seed).run()


def run_adversarial(
    cfg: Config,
    seed: int = 42,
    n_impossible: int = 3,
) -> dict:
    """Try to make the induction loop accept deliberately structureless targets."""
    base = gen_world(
        cfg.schema,
        n_entities=cfg.n_entities,
        density=0.08,
        seed=seed,
    )
    kb = RuleKB(base, cfg.schema, cfg.n_entities)
    rel_names = candidate_relation_names(cfg.schema)
    rng = random.Random(seed + 77)
    attempts: list[dict] = []

    for index in range(n_impossible):
        target = torch.zeros(cfg.n_entities, cfg.n_entities)
        for row in range(cfg.n_entities):
            for col in range(cfg.n_entities):
                if row != col and rng.random() < 0.07:
                    target[row, col] = 1.0

        target_name = f"impossible_{index}"
        if target_name not in kb.program.relations:
            kb.program.relation(target_name, "person", "person")

        pos_pairs = [
            (row, col)
            for row in range(cfg.n_entities)
            for col in range(cfg.n_entities)
            if target[row, col] > 0
        ]
        neg_pairs = [
            (row, col)
            for row in range(cfg.n_entities)
            for col in range(cfg.n_entities)
            if target[row, col] == 0 and row != col
        ]
        rng.shuffle(pos_pairs)
        rng.shuffle(neg_pairs)
        pos = pos_pairs[: cfg.n_pos]
        neg = neg_pairs[: cfg.n_neg]

        if not pos:
            attempts.append(
                {"target": target_name, "outcome": "skipped_no_pos"}
            )
            continue

        proposal = induce_from_examples(
            base,
            pos,
            neg,
            cfg.n_entities,
            max_len=3,
            allowed_rels=rel_names,
        )

        equiv = 0.0
        accepted = False
        outcome = "correctly_rejected"

        if (
            proposal["f1"] >= cfg.f1_threshold
            and proposal["body"] is not None
        ):
            matches = 0
            for world_seed in range(20):
                other_base = gen_world(
                    cfg.schema,
                    n_entities=12,
                    seed=200 + world_seed,
                )
                pred = apply_body(proposal["body"], other_base)
                other_target = torch.zeros(12, 12)
                other_rng = random.Random(seed + world_seed + 300)
                for row in range(12):
                    for col in range(12):
                        if row != col and other_rng.random() < 0.07:
                            other_target[row, col] = 1.0
                if torch.equal(pred, other_target):
                    matches += 1

            equiv = matches / 20
            if equiv >= cfg.min_equiv:
                accepted = kb.add_rule(target_name, proposal["body"])
                outcome = (
                    "FALSIFIED_accepted"
                    if accepted
                    else "rejected_unstable"
                )

        attempts.append(
            {
                "target": target_name,
                "outcome": outcome,
                "best_f1": round(proposal["f1"], 3),
                "equiv": round(equiv, 3),
                "induced": proposal["body"],
                "accepted": accepted,
            }
        )

    return {
        "mode": "adversarial",
        "falsified": any(
            attempt.get("accepted") for attempt in attempts
        ),
        "attempts": attempts,
    }


def verdict(result: dict, cfg: Config) -> tuple[bool, str]:
    """Return exp79-compatible pass/fail semantics for one mode."""
    improvement = result["answered_after"] > result["answered_at_0"]
    accepted_steps = [
        entry
        for entry in result["step_log"]
        if entry["outcome"] == "accepted"
    ]
    all_equiv_ok = all(
        entry["equiv"] >= cfg.min_equiv for entry in accepted_steps
    )
    gen_ok = all(
        sum(entry["gen_scores"].values()) / len(entry["gen_scores"]) >= 0.8
        for entry in accepted_steps
        if entry.get("gen_scores")
    )
    reasons: list[str] = []
    if (
        not improvement
        and result["answered_after"] < result["total_queries"]
    ):
        reasons.append("coverage did not improve")
    if not all_equiv_ok:
        reasons.append(
            f"accepted rule(s) with equiv < {cfg.min_equiv}"
        )
    if not gen_ok:
        reasons.append(
            "poor cross-world generalization (<0.8 on held-out worlds)"
        )
    ok = not reasons
    return ok, "✓ PASS" if ok else f"✗ FAIL — {'; '.join(reasons)}"
