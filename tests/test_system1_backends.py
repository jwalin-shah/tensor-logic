import torch

from tensor_logic.decision_benchmark import (
    DecisionCase,
    DecisionQuestion,
)
from tensor_logic.system1_backends import (
    MiniJevBackend,
    laya_questions,
    normalize_laya_response,
)
from tensor_logic.typed_decision import TensorDecisionModel


def _case():
    return DecisionCase(
        case_id="c1",
        state={"x": 1.0},
        questions=(
            DecisionQuestion(
                "route",
                "choice",
                ("fast", "deep"),
                "Which route?",
            ),
            DecisionQuestion(
                "risk",
                "score",
                ("low", "medium", "high"),
                "Risk level?",
            ),
            DecisionQuestion(
                "escalate",
                "noul",
                ("false", "true"),
                "Escalate?",
            ),
        ),
        targets={
            "route": "fast",
            "risk": "medium",
            "escalate": "false",
        },
    )


def test_laya_question_translation_matches_public_api_shape():
    questions = laya_questions(_case().questions)

    assert questions["route"]["type"] == "choice"
    assert set(questions["route"]["criteria"]) == {"fast", "deep"}
    assert questions["risk"]["criteria"] == ["low", "medium", "high"]
    assert questions["escalate"]["type"] == "noul"


def test_laya_response_normalizes_to_common_schema():
    response = {
        "answers": {
            "route": {
                "type": "choice",
                "choice": "fast",
                "probabilities": {"fast": 0.8, "deep": 0.2},
                "confidence": 0.8,
            },
            "risk": {
                "type": "score",
                "score": 1.1,
                "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
                "confidence": 0.7,
            },
            "escalate": {
                "type": "noul",
                "noul": 0.25,
                "confidence": 0.75,
            },
        }
    }

    rows = normalize_laya_response(
        _case(),
        response,
        latency_ms=12.0,
        model="convaiinnovations/laya",
    )
    by_q = {row.question_id: row for row in rows}

    assert by_q["route"].prediction == "fast"
    assert by_q["risk"].prediction == "medium"
    assert by_q["risk"].probabilities["medium"] == 0.7
    assert by_q["escalate"].prediction == "false"
    assert by_q["escalate"].probabilities["true"] == 0.25
    assert all(row.backend == "laya" for row in rows)


def test_minijev_adapter_uses_same_normalized_result_contract():
    model = TensorDecisionModel(
        2,
        choice_questions={"route": ("fast", "deep")},
        score_questions={"risk": ("low", "medium", "high")},
        noul_questions=("escalate",),
    )
    with torch.no_grad():
        model.choice_weight.zero_()
        model.choice_bias.zero_()
        model.choice_weight[0, 0, 0] = 3.0

        model.score_weight.zero_()
        model.score_bias.zero_()
        model.score_weight[0, 1, 0] = 3.0

        model.noul_weight.zero_()
        model.noul_bias.fill_(-2.0)

    backend = MiniJevBackend(
        model,
        lambda state: torch.tensor([float(state["x"]), 0.0]),
    )

    rows = backend.run(_case())
    by_q = {row.question_id: row for row in rows}

    assert by_q["route"].prediction == "fast"
    assert by_q["risk"].prediction == "medium"
    assert by_q["escalate"].prediction == "false"
    assert all(row.backend == "minijev" for row in rows)
