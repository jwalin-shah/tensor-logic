"""
exp80 — Synthetic Family Validation
====================================
Generates 1000 diverse DependentFamily instances, runs prove_sai() on all of them,
reports distribution stats + edge-case coverage, and outputs a "spot-check sheet"
of 20 representative families formatted for manual FAFSA4caster entry.

Run: uv run python experiments/exp80_validate_synthetic.py

Two goals:
  1. Regression/sanity: SAI formula doesn't crash or produce nonsense for any input
  2. Spot-check sheet: 20 human-readable cases the user can independently verify
     against https://studentaid.gov/aid-estimator/
"""

from __future__ import annotations
import sys, os, random, math, json
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.exp80_fafsa_kb import DependentFamily, prove_sai, SAITrace

random.seed(42)

# ---------------------------------------------------------------------------
# Family generator
# ---------------------------------------------------------------------------

def rand_int(lo: int, hi: int, zero_prob: float = 0.0) -> int:
    if zero_prob > 0 and random.random() < zero_prob:
        return 0
    return random.randint(lo, hi)


def make_family(seed: int | None = None) -> DependentFamily:
    if seed is not None:
        random.seed(seed)

    family_size = random.choices([2, 3, 4, 5, 6, 7], weights=[5, 30, 30, 20, 10, 5])[0]
    num_parents = random.choices([1, 2], weights=[25, 75])[0]
    older_parent_age = random.randint(35, 65)

    # Parent income — wide range: $0 to $300k
    parent_agi = rand_int(0, 300_000)
    # Tax typically 12-22% of AGI, simplified
    eff_rate = random.uniform(0.05, 0.25)
    parent_tax = _ed_round_local(parent_agi * eff_rate)

    # Earned income ≤ AGI
    p1_wages = rand_int(0, parent_agi, zero_prob=0.05)
    p2_wages = rand_int(0, max(0, parent_agi - p1_wages), zero_prob=0.3) if num_parents == 2 else 0

    # Occasional untaxed income
    parent_ira = rand_int(0, 20_000, zero_prob=0.8)
    parent_pension = rand_int(0, 30_000, zero_prob=0.85)
    parent_tex_int = rand_int(0, 5_000, zero_prob=0.85)

    # Parent assets — $0 to $500k
    p_cash = rand_int(0, 100_000, zero_prob=0.1)
    p_inv = rand_int(0, 400_000, zero_prob=0.3)
    p_biz = rand_int(0, 200_000, zero_prob=0.8)
    p_cs = rand_int(0, 15_000, zero_prob=0.85)

    # Student income — typically part-time work
    s_agi = rand_int(0, 30_000, zero_prob=0.3)
    s_wages = rand_int(0, s_agi, zero_prob=0.1)
    s_tax = _ed_round_local(s_agi * random.uniform(0.0, 0.15))

    # Student assets — usually small
    s_cash = rand_int(0, 20_000, zero_prob=0.5)
    s_inv = rand_int(0, 30_000, zero_prob=0.8)

    return DependentFamily(
        parent_agi=parent_agi,
        parent_income_tax_paid=parent_tax,
        parent_earned_income_p1=p1_wages,
        parent_earned_income_p2=p2_wages,
        parent_untaxed_ira_distributions=parent_ira,
        parent_untaxed_pension=parent_pension,
        parent_tax_exempt_interest=parent_tex_int,
        parent_cash_savings=p_cash,
        parent_investment_net_worth=p_inv,
        parent_business_farm_net_worth=p_biz,
        parent_child_support_received=p_cs,
        family_size=family_size,
        num_parents=num_parents,
        older_parent_age=older_parent_age,
        student_agi=s_agi,
        student_income_tax_paid=s_tax,
        student_earned_income=s_wages,
        student_cash_savings=s_cash,
        student_investment_net_worth=s_inv,
    )


def _ed_round_local(x: float) -> int:
    return math.floor(x + 0.5)


# ---------------------------------------------------------------------------
# Edge-case families: zero income, high income, negative SAI, etc.
# ---------------------------------------------------------------------------

NAMED_CASES: list[tuple[str, DependentFamily]] = [
    ("zero-everything", DependentFamily()),
    ("max-pell-floor", DependentFamily(parent_agi=0, family_size=5)),
    ("median-family", DependentFamily(
        parent_agi=85_000, parent_income_tax_paid=10_500,
        parent_earned_income_p1=55_000, parent_earned_income_p2=30_000,
        parent_cash_savings=15_000, parent_investment_net_worth=40_000,
        family_size=4, num_parents=2, older_parent_age=48,
        student_agi=5_000, student_earned_income=5_000, student_income_tax_paid=300,
        student_cash_savings=2_000,
    )),
    ("high-income-no-assets", DependentFamily(
        parent_agi=250_000, parent_income_tax_paid=55_000,
        parent_earned_income_p1=150_000, parent_earned_income_p2=100_000,
        family_size=3, num_parents=2,
    )),
    ("single-parent-low", DependentFamily(
        parent_agi=28_000, parent_income_tax_paid=2_000,
        parent_earned_income_p1=28_000,
        parent_cash_savings=3_000,
        family_size=3, num_parents=1, older_parent_age=40,
    )),
    ("large-family-6", DependentFamily(
        parent_agi=70_000, parent_income_tax_paid=7_000,
        parent_earned_income_p1=40_000, parent_earned_income_p2=30_000,
        parent_cash_savings=20_000, parent_investment_net_worth=50_000,
        family_size=6, num_parents=2,
    )),
    ("big-assets-moderate-income", DependentFamily(
        parent_agi=80_000, parent_income_tax_paid=9_000,
        parent_earned_income_p1=80_000,
        parent_cash_savings=50_000, parent_investment_net_worth=300_000,
        family_size=3, num_parents=1,
    )),
    ("student-has-income", DependentFamily(
        parent_agi=65_000, parent_income_tax_paid=7_000,
        parent_earned_income_p1=65_000,
        family_size=4, num_parents=2,
        student_agi=18_000, student_earned_income=18_000, student_income_tax_paid=1_500,
        student_cash_savings=5_000,
    )),
    ("boundary-pell-threshold", DependentFamily(
        parent_agi=55_000, parent_income_tax_paid=5_000,
        parent_earned_income_p1=35_000, parent_earned_income_p2=20_000,
        parent_cash_savings=10_000,
        family_size=4, num_parents=2,
    )),
    ("very-high-income", DependentFamily(
        parent_agi=500_000, parent_income_tax_paid=140_000,
        parent_earned_income_p1=300_000, parent_earned_income_p2=200_000,
        parent_cash_savings=100_000, parent_investment_net_worth=800_000,
        family_size=3, num_parents=2,
    )),
    ("untaxed-income-heavy", DependentFamily(
        parent_agi=45_000, parent_income_tax_paid=4_000,
        parent_earned_income_p1=45_000,
        parent_untaxed_ira_distributions=20_000, parent_untaxed_pension=15_000,
        parent_tax_exempt_interest=3_000,
        family_size=4, num_parents=2,
    )),
    ("business-owner", DependentFamily(
        parent_agi=120_000, parent_income_tax_paid=22_000,
        parent_earned_income_p1=120_000,
        parent_business_farm_net_worth=400_000,
        parent_cash_savings=30_000,
        family_size=4, num_parents=2,
    )),
    ("age-60-parent", DependentFamily(
        parent_agi=80_000, parent_income_tax_paid=11_000,
        parent_earned_income_p1=80_000,
        parent_cash_savings=200_000, parent_investment_net_worth=150_000,
        family_size=3, num_parents=1, older_parent_age=60,
    )),
    ("negative-sai-likely", DependentFamily(
        parent_agi=10_000, parent_income_tax_paid=500,
        parent_earned_income_p1=10_000,
        family_size=5, num_parents=1,
    )),
    ("very-small-family", DependentFamily(
        parent_agi=100_000, parent_income_tax_paid=15_000,
        parent_earned_income_p1=60_000, parent_earned_income_p2=40_000,
        parent_cash_savings=25_000,
        family_size=2, num_parents=2,
    )),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("  exp80 — FAFSA SAI Synthetic Validation (1000 families)")
    print("=" * 70)

    # Generate 1000 random families
    families_random = [make_family(seed=i) for i in range(1000)]
    families_named = [f for _, f in NAMED_CASES]
    all_families = families_named + families_random

    # Run all through prove_sai
    print(f"\nRunning {len(all_families)} families through prove_sai()...")
    errors = []
    results: list[tuple[DependentFamily, SAITrace]] = []

    for i, fam in enumerate(all_families):
        try:
            trace = prove_sai(fam)
            results.append((fam, trace))
        except Exception as e:
            errors.append((i, fam, str(e)))

    print(f"  Passed: {len(results)}")
    print(f"  Errors: {len(errors)}")
    if errors:
        for idx, fam, err in errors[:5]:
            print(f"    [family {idx}] {err}")

    # Distribution stats
    sais = [t.sai for _, t in results]
    neg = sum(1 for s in sais if s < 0)
    zero = sum(1 for s in sais if s == 0)
    pell_partial = sum(1 for s in sais if 1 <= s <= 7_395)
    above_pell = sum(1 for s in sais if s > 7_395)

    print(f"\n  SAI distribution across {len(sais)} families:")
    print(f"    SAI < 0     (max Pell floor):   {neg:5d}  ({neg/len(sais)*100:.1f}%)")
    print(f"    SAI = 0                         {zero:5d}  ({zero/len(sais)*100:.1f}%)")
    print(f"    1 ≤ SAI ≤ 7395 (partial Pell):  {pell_partial:5d}  ({pell_partial/len(sais)*100:.1f}%)")
    print(f"    SAI > 7395  (no Pell):          {above_pell:5d}  ({above_pell/len(sais)*100:.1f}%)")
    print(f"\n    Min SAI: {min(sais):,}")
    print(f"    Median:  {sorted(sais)[len(sais)//2]:,}")
    print(f"    Max SAI: {max(sais):,}")

    # Check invariants on all results
    print("\n  Checking invariants...")
    inv_fails = 0
    for fam, trace in results:
        step_map = {s.label: s.value for s in trace.steps}
        pc = step_map.get("parent_contribution", 0)
        sci = step_map.get("student_contribution_from_income", 0)
        sca = step_map.get("student_contribution_from_assets", 0)
        expected_sai = max(-1500, pc + sci + sca)
        # allow off-by-1 rounding
        if abs(trace.sai - expected_sai) > 1:
            inv_fails += 1
            if inv_fails <= 3:
                print(f"    FAIL: PC={pc} SCI={sci} SCA={sca} sum={expected_sai} SAI={trace.sai}")

        # SAI must be >= -1500
        if trace.sai < -1500:
            inv_fails += 1
            print(f"    FAIL: SAI={trace.sai} < -1500 floor")

    if inv_fails == 0:
        print(f"  All invariants pass (PC+SCI+SCA=SAI, SAI≥-1500)")
    else:
        print(f"  {inv_fails} invariant failures!")

    # Named case results
    print("\n" + "=" * 70)
    print("  NAMED CASES — use these for manual FAFSA4caster spot-check")
    print("  URL: https://studentaid.gov/aid-estimator/")
    print("=" * 70)
    print()
    for (name, fam), (_, trace) in zip(NAMED_CASES, results[:len(NAMED_CASES)]):
        step_map = {s.label: s.value for s in trace.steps}
        pc = int(step_map.get("parent_contribution", 0))
        sci = int(step_map.get("student_contribution_from_income", 0))
        sca = int(step_map.get("student_contribution_from_assets", 0))
        print(f"  [{name}]")
        print(f"    Family: size={fam.family_size}, parents={fam.num_parents}, parent_age={fam.older_parent_age}")
        print(f"    Parent AGI: ${fam.parent_agi:,}  Tax: ${fam.parent_income_tax_paid:,}")
        print(f"    Parent wages: ${fam.parent_earned_income_p1:,} + ${fam.parent_earned_income_p2:,}")
        if fam.parent_cash_savings or fam.parent_investment_net_worth:
            print(f"    Parent assets: ${fam.parent_cash_savings:,} cash + ${fam.parent_investment_net_worth:,} invest")
        if fam.parent_business_farm_net_worth:
            print(f"    Parent business: ${fam.parent_business_farm_net_worth:,}")
        if fam.student_agi:
            print(f"    Student AGI: ${fam.student_agi:,}  wages: ${fam.student_earned_income:,}")
        if fam.student_cash_savings or fam.student_investment_net_worth:
            print(f"    Student assets: ${fam.student_cash_savings:,} cash + ${fam.student_investment_net_worth:,} invest")
        print(f"    ► Our SAI: {trace.sai:,}  (PC={pc:,} + SCI={sci:,} + SCA={sca:,})")
        print()

    # Save named cases to JSON for reference
    out = []
    for (name, fam), (_, trace) in zip(NAMED_CASES, results[:len(NAMED_CASES)]):
        step_map = {s.label: s.value for s in trace.steps}
        out.append({
            "name": name,
            "inputs": {k: v for k, v in fam.__dict__.items() if v not in (0, False, 3, 2, 45)},
            "sai": trace.sai,
            "parent_contribution": int(step_map.get("parent_contribution", 0)),
            "student_contribution_from_income": int(step_map.get("student_contribution_from_income", 0)),
            "student_contribution_from_assets": int(step_map.get("student_contribution_from_assets", 0)),
        })

    out_path = os.path.join(os.path.dirname(__file__), "exp80_spot_check_cases.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Spot-check cases saved to {out_path}")

    print(f"\n  SUMMARY: {len(results)}/{len(all_families)} families computed without error")
    print(f"  All invariants: {'PASS' if inv_fails == 0 else f'FAIL ({inv_fails} failures)'}")
    print()


if __name__ == "__main__":
    run()
