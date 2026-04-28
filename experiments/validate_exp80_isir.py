"""
Validate exp80 SAI formula against ED's official 2024-25 test ISIRs.

Source: https://github.com/usedgov/fafsa-test-isirs-2024-25
File:   test-isir-files/IDSA25OP-20240308.txt  (100 system-generated records)

Checks (Formula A = dependent student records only):
  1. _aai_to_parent_contribution(PAAI) == ISIR Parent Contribution
  2. PC + SCI + SCA == SAI  (formula summation integrity)
  3. IPA values in ISIRs match our IPA_TABLE

Run: uv run python experiments/validate_exp80_isir.py <path-to-IDSA25OP-20240308.txt>
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp80_fafsa_kb import _ipa, _aai_to_parent_contribution, IPA_TABLE

IPA_TO_FAM = {v: k for k, v in IPA_TABLE.items()}

FIELDS = {
    'sai':  (175, 181),
    'formula': (187, 188),
    'ipa':  (2895, 2910),
    'eea':  (2910, 2925),
    'paai': (2940, 2955),
    'pc':   (2955, 2970),
    'sci':  (3060, 3075),
    'sca':  (3162, 3174),
    'fam':  (3177, 3180),
}


def pi(line, key):
    s, e = FIELDS[key]
    v = line[s:e].strip()
    return int(v) if v else None


def validate(isir_path: str):
    with open(isir_path) as f:
        lines = [l.rstrip('\n') for l in f]

    passed = failed = skipped = 0
    failures = []

    for lineno, line in enumerate(lines, 1):
        if len(line) < 3200 or line[187:188] != 'A':
            continue

        sai  = pi(line, 'sai')
        ipa  = pi(line, 'ipa')
        paai = pi(line, 'paai')
        pc   = pi(line, 'pc')
        sci  = pi(line, 'sci')
        sca  = pi(line, 'sca')

        if paai is None or pc is None or ipa is None:
            skipped += 1
            continue

        our_pc = _aai_to_parent_contribution(paai)
        pc_ok = (our_pc == pc)

        sai_ok = True
        if sai is not None and sci is not None and sca is not None:
            sai_ok = (pc + sci + sca == sai)

        ipa_ok = ipa in IPA_TO_FAM

        if pc_ok and sai_ok and ipa_ok:
            passed += 1
        else:
            failed += 1
            failures.append((lineno, ipa, paai, pc, our_pc, sai, sci, sca, pc_ok, sai_ok, ipa_ok))

    print(f"Formula A records: {passed+failed+skipped}")
    print(f"  PASS={passed}  FAIL={failed}  SKIP={skipped}")

    if failures:
        print("\nFailing records:")
        for row in failures:
            lineno, ipa, paai, pc, our_pc, sai, sci, sca, pc_ok, sai_ok, ipa_ok = row
            print(f"  line={lineno} IPA={ipa} PAAI={paai} ISIR_PC={pc} OUR_PC={our_pc} "
                  f"SAI={sai} pc_ok={pc_ok} sai_ok={sai_ok} ipa_ok={ipa_ok}")
        return False

    print("\nAll checks PASS — AAI formula, SAI summation, and IPA table verified against ED test ISIRs.")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else (
        os.path.expanduser("~/tmp/fafsa-test-isirs-2024-25/test-isir-files/IDSA25OP-20240308.txt")
    )
    if not os.path.exists(path):
        print(f"ISIR file not found: {path}")
        print("Clone https://github.com/usedgov/fafsa-test-isirs-2024-25 and pass path as argument.")
        sys.exit(1)
    ok = validate(path)
    sys.exit(0 if ok else 1)
