from tensor_logic.universal_context import (
    reconstruct_generic_record,
    tensorize_generic_context,
)


def _context():
    return {
        "schema_version": "lifeops.context.v1",
        "checked_at": "2026-09-20T00:00:00Z",
        "read_only": True,
        "limitations": ["stale_source"],
        "sections": {
            "people": [
                {
                    "item_id": "person:1",
                    "title": "Ada",
                    "score": 0.75,
                    "active": True,
                    "nickname": None,
                    "tags": ["friend", "research"],
                    "metadata": {
                        "a.b": 3,
                        "x[y]": "brackets",
                        "tilde~key": "tilde",
                        "empty": {},
                    },
                }
            ],
            "projects": [],
        },
        "source_health": {
            "providers": {
                "status": "degraded",
                "providers": [
                    {
                        "provider": "gmail",
                        "readable": False,
                    }
                ],
            }
        },
        "provenance": {
            "reference_count": 1,
        },
    }


def test_generic_context_round_trips_arbitrary_nested_record():
    world = tensorize_generic_context(_context())

    reconstructed = reconstruct_generic_record(
        world,
        "people:person:1",
    )
    assert reconstructed == _context()["sections"]["people"][0]


def test_generic_context_preserves_root_and_empty_sections():
    world = tensorize_generic_context(_context())

    root = reconstruct_generic_record(world, "context:root")
    assert root["schema_version"] == "lifeops.context.v1"
    assert root["read_only"] is True
    assert root["limitations"] == ["stale_source"]

    assert "projects" not in {
        coordinate[0]
        for coordinate in world.tensors["section_record"].coordinates()
    }


def test_generic_projection_keeps_scalar_types_distinct():
    world = tensorize_generic_context(_context())
    record_id = "people:person:1"

    types = {
        (path, scalar_type)
        for rec, path, scalar_type in world.tensors[
            "scalar_type"
        ].coordinates()
        if rec == record_id
    }

    assert ("$.score", "float") in types
    assert ("$.active", "bool") in types
    assert ("$.nickname", "null") in types
    assert ("$.metadata.a~1b", "int") in types
    assert ("$.metadata.x~2y~3", "string") in types
    assert ("$.metadata.tilde~0key", "string") in types


def test_generic_projection_is_reproducible():
    assert (
        tensorize_generic_context(_context()).digest
        == tensorize_generic_context(_context()).digest
    )
