"""
FAFSA SAI Wizard — ask plain-English questions, compute SAI, explain each step.

Run: uv run python experiments/exp80_fafsa_wizard.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp80_fafsa_kb import DependentFamily, prove_sai, SAITrace


# ---------------------------------------------------------------------------
# Plain-English question helpers
# ---------------------------------------------------------------------------

def ask(prompt: str, default=None, cast=int, allow_blank=True) -> int | None:
    hint = f"  (press Enter for {default:,})" if default is not None else ""
    while True:
        raw = input(f"\n  {prompt}{hint}\n  > ").strip()
        if not raw:
            if default is not None:
                return default
            if allow_blank:
                return 0
        try:
            return cast(raw.replace(",", "").replace("$", ""))
        except ValueError:
            print("  Please enter a whole dollar amount (no decimals).")


def ask_yn(prompt: str, default: bool = False) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    raw = input(f"\n  {prompt} {hint}\n  > ").strip().lower()
    if not raw:
        return default
    return raw.startswith("y")


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def note(text: str):
    for line in text.strip().split("\n"):
        print(f"  ℹ  {line.strip()}")


# ---------------------------------------------------------------------------
# Explanation strings for each SAI component
# ---------------------------------------------------------------------------

EXPLANATIONS = {
    "parent_income_tax_paid": (
        "Tax paid",
        "The federal income taxes your parents actually paid last year. "
        "This reduces what the formula counts as available income."
    ),
    "parent_medicare_allowance": (
        "Payroll taxes (Medicare)",
        "Medicare and Social Security taxes your parents paid on wages. "
        "Also subtracted from available income — you already paid this, "
        "it's not money you can spend on college."
    ),
    "parent_oasdi_allowance": (
        "Payroll taxes (Social Security)",
        "The Social Security portion of payroll taxes. Same idea — money already gone."
    ),
    "parent_income_protection_allowance": (
        "Income Protection Allowance (IPA)",
        "The big one. A fixed amount the formula shields from assessment — "
        "it's meant to cover basic living costs for a family your size. "
        "The bigger your family, the larger this shield."
    ),
    "parent_employment_expense_allowance": (
        "Employment Expense Allowance (EEA)",
        "If both parents (or a single working parent) earn income, the formula "
        "gives a small allowance — up to $4,730 — recognizing that two incomes "
        "means extra costs like childcare."
    ),
    "parent_available_income": (
        "Parent Available Income (PAI)",
        "What's left of parent income after all the allowances above. "
        "This is the income the formula thinks you could put toward college."
    ),
    "parent_contribution_from_assets": (
        "Parent Contribution from Assets (PCA)",
        "12% of your parents' net assets (savings, investments, business equity). "
        "The formula counts assets as if you could convert 12% of them annually."
    ),
    "parent_adjusted_available_income": (
        "Parent Adjusted Available Income (PAAI)",
        "Available income plus the asset contribution. "
        "This is the combined measure of your parents' financial strength."
    ),
    "parent_contribution": (
        "Parent Contribution (PC)",
        "The actual dollar amount your parents are expected to contribute, "
        "calculated using a progressive rate schedule (22–47%) on PAAI. "
        "Higher PAAI → higher rate, similar to income tax brackets."
    ),
    "student_contribution_from_income": (
        "Student Contribution from Income (SCI)",
        "50% of your available income after your own allowances ($11,130 IPA + "
        "payroll taxes + income tax). Student income is assessed more heavily "
        "than parent income. Can be negative — down to −$1,500."
    ),
    "student_contribution_from_assets": (
        "Student Contribution from Assets (SCA)",
        "20% of your own assets (savings, investments). "
        "Student assets are assessed at 20% vs 12% for parents."
    ),
    "student_aid_index": (
        "Student Aid Index (SAI)",
        "The final number. Lower = more need = more aid. "
        "SAI can be negative (minimum −1,500). "
        "Your school subtracts SAI from Cost of Attendance to get your financial need."
    ),
}


def explain_trace(trace: SAITrace):
    print(f"\n{'═'*60}")
    print("  HOW YOUR SAI WAS CALCULATED")
    print(f"{'═'*60}")

    # Group steps into phases
    phases = [
        ("PARENT INCOME", [
            "parent_income_additions", "parent_income_offsets", "parent_total_income",
            "parent_income_tax_paid",
            "parent_medicare_allowance", "parent_oasdi_allowance",
            "parent_income_protection_allowance", "parent_employment_expense_allowance",
            "parent_total_allowances", "parent_available_income",
        ]),
        ("PARENT ASSETS", [
            "parent_business_farm_adjusted_nw", "parent_net_worth",
            "parent_asset_protection_allowance", "parent_contribution_from_assets",
        ]),
        ("PARENT CONTRIBUTION", [
            "parent_adjusted_available_income", "parent_contribution",
        ]),
        ("STUDENT INCOME", [
            "student_income_additions", "student_income_offsets", "student_total_income",
            "student_medicare_allowance", "student_oasdi_allowance",
            "parents_negative_paai_allowance", "student_total_allowances",
            "student_available_income", "student_contribution_from_income",
        ]),
        ("STUDENT ASSETS", [
            "student_business_farm_adjusted_nw", "student_net_worth",
            "student_contribution_from_assets",
        ]),
        ("FINAL SAI", ["student_aid_index"]),
    ]

    step_map = {s.label: s for s in trace.steps}

    for phase_name, labels in phases:
        relevant = [step_map[l] for l in labels if l in step_map and step_map[l].value != 0
                    or l in ("parent_available_income", "parent_adjusted_available_income",
                             "parent_contribution", "student_contribution_from_income",
                             "student_aid_index")]
        relevant = [step_map[l] for l in labels if l in step_map]
        if not any(s.value != 0 for s in relevant) and phase_name not in ("PARENT CONTRIBUTION", "FINAL SAI"):
            continue

        print(f"\n  ── {phase_name} ──")
        for s in relevant:
            if s.value == 0 and s.label not in (
                "parent_total_income", "parent_available_income",
                "parent_adjusted_available_income", "parent_contribution",
                "student_contribution_from_income", "student_aid_index",
                "student_total_income", "student_available_income",
            ):
                continue  # skip zero non-key lines to reduce clutter

            short, explanation = EXPLANATIONS.get(s.label, (s.label.replace("_", " ").title(), ""))
            sign = "+" if s.value > 0 else ""
            print(f"\n    {short}: ${s.value:,}")
            if explanation:
                # Word-wrap to 54 chars
                words = explanation.split()
                line, lines = [], []
                for w in words:
                    if sum(len(x)+1 for x in line) + len(w) > 54:
                        lines.append(" ".join(line))
                        line = [w]
                    else:
                        line.append(w)
                if line:
                    lines.append(" ".join(line))
                for l in lines:
                    print(f"    → {l}")

    # Bottom line
    print(f"\n{'═'*60}")
    sai = trace.sai
    print(f"  YOUR SAI:  {sai:,}")
    if sai <= 0:
        print("  → You likely qualify for a maximum Federal Pell Grant.")
    elif sai <= 7_395:
        print("  → You likely qualify for a partial Pell Grant.")
    else:
        print("  → Your SAI exceeds the Pell Grant range.")
        print("    You may still qualify for loans and school-based aid.")
    print(f"{'═'*60}")


# ---------------------------------------------------------------------------
# Main questionnaire
# ---------------------------------------------------------------------------

def run():
    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║   FAFSA Student Aid Index (SAI) Estimator           ║")
    print("  ║   2024-25 Award Year · Formula A (Dependent)        ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    note(
        "This estimates your SAI using the official ED formula.\n"
        "All figures are for the 2022 tax year (prior-prior year).\n"
        "Press Enter to use 0 for any question that doesn't apply."
    )

    # ── FAMILY STRUCTURE ──────────────────────────────────────────
    section("ABOUT YOUR FAMILY")

    family_size = ask(
        "How many people are in your household? "
        "(Include parents, you, and any siblings they support)",
        default=3, cast=int
    )
    note(f"Family of {family_size} → your Income Protection Allowance shields more income from assessment.")

    num_parents = ask(
        "How many parents/contributors are in the household? (1 or 2)",
        default=2, cast=int
    )
    num_parents = max(1, min(2, num_parents))

    older_parent_age = ask(
        "What is the age of the older parent as of December 31, 2023?",
        default=45, cast=int
    )

    # ── PARENT INCOME ─────────────────────────────────────────────
    section("PARENT INCOME  (2022 tax year)")

    note(
        "We need your parents' 2022 income figures.\n"
        "Most of this comes straight from their 2022 federal tax return."
    )

    parent_agi = ask("Parents' Adjusted Gross Income (AGI) — Form 1040, Line 11", default=0)
    note("AGI = total income minus above-the-line deductions like student loan interest, IRA contributions, etc.")

    parent_tax = ask("Federal income tax paid — Form 1040, Line 24 minus Line 31", default=0)

    # Earned income (for payroll taxes and EEA)
    if num_parents == 2:
        note("We need each parent's wages separately to compute payroll taxes correctly.")
        p1_wages = ask("Parent 1 wages / salary (W-2 Box 1, or Schedule C net if self-employed)", default=0)
        p2_wages = ask("Parent 2 wages / salary", default=0)
    else:
        p1_wages = ask("Parent wages / salary", default=0)
        p2_wages = 0

    # Less common income additions
    has_extras = ask_yn(
        "Did your parents have any untaxed income? "
        "(IRA distributions, pension distributions, tax-exempt interest, foreign income exclusion)",
        default=False
    )
    ira_dist = tax_exempt = pension = foreign = 0
    if has_extras:
        ira_dist  = ask("Untaxed IRA distributions (minus any rollover; enter 0 if none)", default=0)
        pension   = ask("Untaxed pension/annuity distributions (minus any rollover; enter 0 if none)", default=0)
        tax_exempt = ask("Tax-exempt interest income", default=0)
        foreign   = ask("Foreign income exclusion (from US tax return)", default=0)

    # Income offsets
    has_offsets = ask_yn(
        "Did your parents report any college grant/scholarship aid as taxable income, "
        "education credits, or federal work-study earnings?",
        default=False
    )
    scholarships = edu_credits = work_study = 0
    if has_offsets:
        scholarships  = ask("Taxable college grant/scholarship aid reported as income", default=0)
        edu_credits   = ask("Education credits (American Opportunity / Lifetime Learning)", default=0)
        work_study    = ask("Federal Work-Study earnings", default=0)

    # ── PARENT ASSETS ─────────────────────────────────────────────
    section("PARENT ASSETS  (current balances)")

    note(
        "Assets are assessed at 12% per year — so $100,000 in assets\n"
        "adds about $12,000 to your SAI. Primary home is NOT counted."
    )

    p_cash        = ask("Cash, savings, and checking accounts", default=0)
    p_investments = ask(
        "Net worth of investments (stocks, bonds, real estate excl. primary home, "
        "529 plans, mutual funds, etc.) — enter 0 if negative",
        default=0
    )
    p_business    = ask(
        "Net worth of small businesses or investment farms you own "
        "(enter 0 if none or if it's your primary employer with < 100 employees)",
        default=0
    )
    p_child_support = ask("Child support received in the last calendar year", default=0)

    # ── STUDENT INCOME ────────────────────────────────────────────
    section("YOUR INCOME (the student, 2022 tax year)")

    note(
        "Student income is assessed at 50% — much heavier than parents.\n"
        "But you get a $11,130 shield (student IPA) before that kicks in."
    )

    s_agi         = ask("Your (student) Adjusted Gross Income", default=0)
    s_tax         = ask("Your federal income tax paid", default=0)
    s_wages       = ask("Your wages / earned income (for payroll tax calculation)", default=0)

    has_s_extras  = ask_yn("Any untaxed student income (IRA, pension distributions, etc.)?", default=False)
    s_ira = s_pension = s_tax_exempt = s_foreign = 0
    if has_s_extras:
        s_ira       = ask("Student untaxed IRA distributions (minus rollover)", default=0)
        s_pension   = ask("Student untaxed pension distributions (minus rollover)", default=0)
        s_tax_exempt = ask("Student tax-exempt interest", default=0)
        s_foreign   = ask("Student foreign income exclusion", default=0)

    has_s_offsets = ask_yn(
        "Do you have any college grant/scholarship aid reported as income, "
        "education credits, or work-study?",
        default=False
    )
    s_scholarships = s_edu_credits = s_work_study = 0
    if has_s_offsets:
        s_scholarships = ask("Taxable scholarship/grant aid reported as income", default=0)
        s_edu_credits  = ask("Education credits", default=0)
        s_work_study   = ask("Federal Work-Study earnings", default=0)

    # ── STUDENT ASSETS ────────────────────────────────────────────
    section("YOUR ASSETS (the student)")

    note("Student assets are assessed at 20% — heavier than the 12% for parents.")

    s_cash        = ask("Your cash, savings, and checking accounts", default=0)
    s_investments = ask("Your investments (enter 0 if negative)", default=0)
    s_business    = ask("Your business/farm net worth (enter 0 if none)", default=0)

    # ── COMPUTE ───────────────────────────────────────────────────
    family = DependentFamily(
        parent_agi=parent_agi,
        parent_deductible_ira_payments=0,
        parent_tax_exempt_interest=tax_exempt,
        parent_untaxed_ira_distributions=ira_dist,
        parent_untaxed_pension=pension,
        parent_foreign_income_exclusion=foreign,
        parent_taxable_scholarships=scholarships,
        parent_education_credits=edu_credits,
        parent_work_study=work_study,
        parent_income_tax_paid=parent_tax,
        parent_earned_income_p1=p1_wages,
        parent_earned_income_p2=p2_wages,
        parent_cash_savings=p_cash,
        parent_investment_net_worth=p_investments,
        parent_business_farm_net_worth=p_business,
        parent_child_support_received=p_child_support,
        family_size=family_size,
        num_parents=num_parents,
        older_parent_age=older_parent_age,
        student_agi=s_agi,
        student_tax_exempt_interest=s_tax_exempt,
        student_untaxed_ira_distributions=s_ira,
        student_untaxed_pension=s_pension,
        student_foreign_income_exclusion=s_foreign,
        student_taxable_scholarships=s_scholarships,
        student_education_credits=s_edu_credits,
        student_work_study=s_work_study,
        student_income_tax_paid=s_tax,
        student_earned_income=s_wages,
        student_cash_savings=s_cash,
        student_investment_net_worth=s_investments,
        student_business_farm_net_worth=s_business,
    )

    print("\n  Computing...")
    trace = prove_sai(family)
    explain_trace(trace)

    # Counterfactual: what if parent income dropped $10k?
    if parent_agi >= 10_000:
        cf = prove_sai(DependentFamily(**{**family.__dict__, "parent_agi": parent_agi - 10_000}))
        delta = cf.sai - trace.sai
        print(f"\n  WHAT-IF: If parent AGI were $10,000 lower → SAI would be {cf.sai:,} "
              f"({'+' if delta >= 0 else ''}{delta:,})")


if __name__ == "__main__":
    run()
