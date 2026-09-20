from tensor_logic.lifeops_adapter import tensorize_life_context


def _fixture():
    return {
        "schema_version": "lifeops.context.v1",
        "sections": {
            "attention": [
                {
                    "item_id": "thread:1",
                    "source": "gmail",
                    "state": "READY_HUMAN",
                    "attention_class": "now",
                    "reason": "Reply requested",
                    "source_ref": {
                        "kind": "thread",
                        "source": "gmail",
                        "thread_id": "1",
                    },
                    "attribution": {
                        "authority": "gmail",
                        "derived": True,
                        "read_only": True,
                        "method": "inbox_now_rule_projection",
                    },
                    "details": {
                        "participants": ["person-a"],
                        "last_message_at": "2026-09-20T01:00:00Z",
                        "needs_reply": True,
                        "rank": 0.8,
                    },
                    "category": "reply_now",
                }
            ],
            "people": [
                {
                    "item_id": "person:me",
                    "source": "contacts",
                }
            ],
        },
        "source_health": {
            "providers": {
                "providers": [
                    {
                        "provider": "google_gmail",
                        "readable": False,
                        "syncable": False,
                        "writable": False,
                        "blockers": ["service_not_loaded"],
                    },
                    {
                        "provider": "imessage",
                        "readable": True,
                        "syncable": False,
                        "writable": False,
                        "blockers": [],
                    },
                ]
            }
        },
    }


def test_life_context_attention_is_candidate_not_canonical():
    world = tensorize_life_context(_fixture())

    assert world.tensors["candidate_needs_reply"].get(
        ("thread:1",)
    ) == 1.0
    provenance = world.tensors[
        "candidate_needs_reply"
    ].provenance(("thread:1",))
    assert provenance.metadata["epistemic_status"] == "candidate"
    assert provenance.metadata["derived"] is True
    assert provenance.source_refs == ("gmail",)


def test_life_context_preserves_participant_and_rank():
    world = tensorize_life_context(_fixture())

    assert world.tensors["attention_participant"].get(
        ("participant:person-a", "thread:1")
    ) == 1.0
    assert world.tensors["candidate_rank"].get(
        ("thread:1",)
    ) == 0.8


def test_source_health_is_tensorized_without_promoting_attention():
    world = tensorize_life_context(_fixture())

    assert world.tensors["provider_readable"].get(
        ("google_gmail",)
    ) == 0.0
    assert world.tensors["provider_readable"].get(
        ("imessage",)
    ) == 1.0


def test_lifeops_transport_tensor_digest_is_reproducible():
    assert (
        tensorize_life_context(_fixture()).digest
        == tensorize_life_context(_fixture()).digest
    )
