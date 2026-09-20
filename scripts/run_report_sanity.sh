#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-/tmp/tensor-logic-report-v1}"
mkdir -p "$OUT_DIR"

python -m pytest   tests/test_actr_memory.py   tests/test_metacognition.py   tests/test_provenance_semiring.py   tests/test_provenance_datalog.py   tests/test_kernel_contract.py   tests/test_reasoning_persistence.py   tests/test_software_graph.py   tests/test_temporal_execution_graph.py   tests/test_predictive_world.py   tests/test_rule_hypothesis_eval.py   tests/test_world_event_log.py   tests/test_exp105_hybrid_world.py   -q | tee "$OUT_DIR/pytest-report-experiments.txt"

python experiments/exp101_actr_retrieval.py   --out "$OUT_DIR/exp101-actr.json"

python experiments/exp102_metacognition.py   --out "$OUT_DIR/exp102-metacognition.json"

python experiments/exp103_provenance.py   --out "$OUT_DIR/exp103-provenance.json"

python experiments/exp104_kernel_contract.py   --out "$OUT_DIR/exp104-kernel.json"

python experiments/exp105_hybrid_software_world.py   --out "$OUT_DIR/exp105-hybrid-world.json"

python - <<'PY' "$OUT_DIR"
import hashlib
import json
import pathlib
import platform
import sys
import torch

root = pathlib.Path(sys.argv[1])
files = sorted(root.glob("*.json"))
manifest = {
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    },
    "artifacts": {},
}
for path in files:
    raw = path.read_bytes()
    manifest["artifacts"][path.name] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
(root / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True)
)
print((root / "manifest.json").read_text())
PY

echo "report experiment manifests: $OUT_DIR"
