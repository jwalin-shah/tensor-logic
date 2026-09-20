"""exp99: reproducible CPU performance benchmark for Personal Physics tensors."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import torch

from tensor_logic.cognitive_program import CognitiveProgramTrace
from tensor_logic.tensor_ops import (
    sparse_binary_compose,
    topk_indices,
    vectorized_weighted_score,
)
from tensor_logic.universal_context import tensorize_generic_context
from tensor_logic.world_tensor import TensorWorld


def timed(fn, repeats=1):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "repeats": repeats,
    }, result


def build_sparse_world():
    world = TensorWorld()
    people = tuple(f"p{i}" for i in range(1000))
    events = tuple(f"e{i}" for i in range(10000))
    topics = tuple(f"t{i}" for i in range(200))

    world.add_axis("Person", "Person", people)
    world.add_axis("Event", "Event", events)
    world.add_axis("Topic", "Topic", topics)

    attends = world.add_tensor(
        "attends",
        ("Person", "Event"),
    )
    event_topic = world.add_tensor(
        "event_topic",
        ("Event", "Topic"),
    )

    for event_index, event in enumerate(events):
        for offset in range(10):
            person_index = (
                event_index * 17 + offset * 101
            ) % len(people)
            attends.set(
                (people[person_index], event),
                1.0,
            )
        event_topic.set(
            (event, topics[event_index % len(topics)]),
            1.0,
        )

    return world, attends, event_topic


def benchmark_point_lookup(attends):
    coordinates = attends.coordinates()
    sample = [
        coordinates[(i * 7919) % len(coordinates)]
        for i in range(10000)
    ]

    def run():
        total = 0.0
        for _ in range(10):
            for coordinate in sample:
                total += attends.get(coordinate)
        return total

    timing, total = timed(run, repeats=3)
    if total != 100000.0:
        raise AssertionError(f"unexpected lookup total {total}")
    timing["lookups"] = 100000
    timing["lookups_per_second"] = (
        100000.0 / (timing["median_ms"] / 1000.0)
    )
    timing["microseconds_per_lookup"] = (
        timing["median_ms"] * 1000.0 / 100000.0
    )
    return timing


def benchmark_sparse_compose(attends, event_topic):
    materialize_timing, pair = timed(
        lambda: (attends.sparse(), event_topic.sparse()),
        repeats=3,
    )
    left, right = pair

    pure_timing, pure_product = timed(
        lambda: torch.sparse.mm(left, right).coalesce(),
        repeats=5,
    )
    api_timing, api_product = timed(
        lambda: sparse_binary_compose(
            attends,
            event_topic,
        ),
        repeats=3,
    )
    if pure_product.shape != torch.Size([1000, 200]):
        raise AssertionError("unexpected composition shape")
    if api_product._nnz() != pure_product._nnz():
        raise AssertionError("API and pure sparse products differ in nnz")

    return {
        "input_nnz": {
            "attends": attends.sparse()._nnz(),
            "event_topic": event_topic.sparse()._nnz(),
        },
        "output_shape": list(pure_product.shape),
        "output_nnz": pure_product._nnz(),
        "materialize": materialize_timing,
        "pure_sparse_mm": pure_timing,
        "api_path_including_materialization": api_timing,
    }


def benchmark_scoring():
    candidate_count = 10000
    base = torch.arange(
        candidate_count,
        dtype=torch.float32,
    )
    features = torch.stack(
        (
            (base % 101) / 100.0,
            (base % 79) / 78.0,
            (base % 2),
            (base % 53) / 52.0,
        ),
        dim=1,
    )
    weights = torch.tensor(
        [0.25, 0.15, 0.45, 0.15],
        dtype=torch.float32,
    )

    def score_many():
        result = None
        for _ in range(200):
            result = vectorized_weighted_score(
                features,
                weights,
            )
        return result

    score_timing, scores = timed(score_many, repeats=5)
    score_timing["batches"] = 200
    score_timing["candidates_per_batch"] = candidate_count
    score_timing["median_ms_per_10k_candidates"] = (
        score_timing["median_ms"] / 200.0
    )

    def topk_many():
        result = None
        for _ in range(100):
            result = topk_indices(scores, 25)
        return result

    topk_timing, top = timed(topk_many, repeats=5)
    topk_timing["batches"] = 100
    topk_timing["median_ms_per_topk_10k"] = (
        topk_timing["median_ms"] / 100.0
    )
    values, indices = top
    if len(values) != 25 or len(indices) != 25:
        raise AssertionError("top-k result has wrong size")

    return {
        "candidate_count": candidate_count,
        "feature_count": 4,
        "score": score_timing,
        "topk": topk_timing,
    }


def build_generic_context(record_count=1000):
    attention = []
    for i in range(record_count):
        attention.append(
            {
                "item_id": f"item:{i}",
                "source": "gmail" if i % 2 == 0 else "imessage",
                "state": "READY_HUMAN",
                "attention_class": "now" if i % 3 else "waiting",
                "category": "reply_now" if i % 4 else "task",
                "details": {
                    "participants": [
                        f"person:{i % 100}",
                        f"person:{(i + 1) % 100}",
                    ],
                    "needs_reply": bool(i % 2),
                    "rank": (i % 10) / 10.0,
                    "last_message_at": f"2026-09-{1 + i % 20:02d}T12:00:00Z",
                },
                "source_ref": {
                    "kind": "thread",
                    "source": "gmail",
                    "thread_id": str(i),
                },
                "attribution": {
                    "authority": "gmail",
                    "derived": True,
                    "read_only": True,
                    "method": "synthetic_projection",
                },
            }
        )
    return {
        "schema_version": "lifeops.context.v1",
        "checked_at": "2026-09-20T00:00:00Z",
        "read_only": True,
        "sections": {
            "attention": attention,
            "people": [],
            "places": [],
            "projects": [],
            "goals": [],
            "decisions": [],
            "notes": [],
            "documents": [],
            "commitments": [],
        },
        "source_health": {
            "providers": {
                "status": "ok",
                "providers": [],
            }
        },
        "provenance": {
            "reference_count": record_count,
        },
    }


def benchmark_generic_context():
    context = build_generic_context(1000)
    timing, world = timed(
        lambda: tensorize_generic_context(context),
        repeats=3,
    )
    timing["records"] = 1000
    timing["tensor_count"] = len(world.tensors)
    timing["axis_count"] = len(world.axes)
    timing["world_digest"] = world.digest
    timing["total_tensor_coordinates"] = sum(
        len(tensor.coordinates())
        for tensor in world.tensors.values()
    )
    return timing


def benchmark_incremental_updates(world, attends, event_topic):
    new_events = tuple(f"e{i}" for i in range(10000, 11000))

    start = time.perf_counter()
    world.extend_axis("Event", new_events)
    extend_ms = (time.perf_counter() - start) * 1000.0

    people = world.axes["Person"].symbols
    topics = world.axes["Topic"].symbols

    start = time.perf_counter()
    inserted = 0
    for local_index, event in enumerate(new_events):
        event_index = 10000 + local_index
        for offset in range(10):
            person_index = (
                event_index * 17 + offset * 101
            ) % len(people)
            attends.set(
                (people[person_index], event),
                1.0,
            )
            inserted += 1
        event_topic.set(
            (event, topics[event_index % len(topics)]),
            1.0,
        )
        inserted += 1
    insert_ms = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    retracted = 0
    for event_index in range(100):
        event = f"e{event_index}"
        for offset in range(10):
            person_index = (
                event_index * 17 + offset * 101
            ) % len(people)
            if attends.remove((people[person_index], event)):
                retracted += 1
    retract_ms = (time.perf_counter() - start) * 1000.0

    return {
        "new_axis_symbols": len(new_events),
        "inserted_coordinates": inserted,
        "retracted_coordinates": retracted,
        "axis_extend_ms": extend_ms,
        "insert_ms": insert_ms,
        "retract_ms": retract_ms,
        "inserts_per_second": (
            inserted / (insert_ms / 1000.0)
            if insert_ms > 0
            else None
        ),
        "retractions_per_second": (
            retracted / (retract_ms / 1000.0)
            if retract_ms > 0
            else None
        ),
        "final_attends_nnz": attends.nnz,
        "final_event_topic_nnz": event_topic.nnz,
    }


def benchmark_cognitive_trace():
    def build():
        trace = CognitiveProgramTrace("benchmark-trace")
        trace.add_artifact(
            "root",
            kind="observation",
            payload={"seed": 1},
            source_refs=("benchmark",),
        )
        previous_artifact = "root"
        previous_step = None
        for i in range(500):
            output = f"artifact:{i}"
            trace.add_artifact(
                output,
                kind="derived",
                payload={"i": i, "value": i * 2},
            )
            operator = "DERIVE" if i % 2 == 0 else "SCORE"
            kwargs = {}
            if previous_step is not None:
                kwargs["parent_step_ids"] = (previous_step,)
            trace.add_step(
                f"step:{i}",
                operator=operator,
                operator_id="benchmark.operator",
                operator_version="1",
                input_artifact_ids=(previous_artifact,),
                output_artifact_ids=(output,),
                **kwargs,
            )
            previous_artifact = output
            previous_step = f"step:{i}"
        return trace

    build_timing, trace = timed(build, repeats=3)
    digest_timing, digest = timed(
        lambda: trace.process_digest,
        repeats=10,
    )
    return {
        "steps": 500,
        "artifacts": 501,
        "build": build_timing,
        "digest": digest_timing,
        "process_digest": digest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="experiments/exp99_tensor_performance_data/results.json",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)

    world_build_timing, built = timed(
        build_sparse_world,
        repeats=1,
    )
    world, attends, event_topic = built

    result = {
        "experiment": "exp99_tensor_performance",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "scale": {
            "people": 1000,
            "events": 10000,
            "topics": 200,
            "sparse_coordinates": (
                len(attends.coordinates())
                + len(event_topic.coordinates())
            ),
        },
        "world_build": {
            **world_build_timing,
            "world_digest": world.digest,
        },
        "point_lookup": benchmark_point_lookup(attends),
        "sparse_composition": benchmark_sparse_compose(
            attends,
            event_topic,
        ),
        "attention_scoring": benchmark_scoring(),
        "generic_context": benchmark_generic_context(),
        "cognitive_trace": benchmark_cognitive_trace(),
        "incremental_updates": benchmark_incremental_updates(
            world,
            attends,
            event_topic,
        ),
        "process_max_rss_kb": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))

    print("exp99: tensor performance")
    print(
        "build 110k sparse coordinates: "
        f"{result['world_build']['median_ms']:.2f} ms"
    )
    print(
        "point lookup: "
        f"{result['point_lookup']['lookups_per_second']:.0f}/s "
        f"({result['point_lookup']['microseconds_per_lookup']:.3f} us)"
    )
    print(
        "sparse A@B pure: "
        f"{result['sparse_composition']['pure_sparse_mm']['median_ms']:.2f} ms"
    )
    print(
        "sparse A@B API incl materialization: "
        f"{result['sparse_composition']['api_path_including_materialization']['median_ms']:.2f} ms"
    )
    print(
        "score 10k candidates: "
        f"{result['attention_scoring']['score']['median_ms_per_10k_candidates']:.4f} ms"
    )
    print(
        "top25 of 10k: "
        f"{result['attention_scoring']['topk']['median_ms_per_topk_10k']:.4f} ms"
    )
    print(
        "generic 1k-record context tensorization: "
        f"{result['generic_context']['median_ms']:.2f} ms"
    )
    print(
        "500-step cognitive trace build: "
        f"{result['cognitive_trace']['build']['median_ms']:.2f} ms"
    )
    print(
        "500-step trace digest: "
        f"{result['cognitive_trace']['digest']['median_ms']:.2f} ms"
    )
    print(
        "incremental 11k inserts: "
        f"{result['incremental_updates']['insert_ms']:.2f} ms "
        f"({result['incremental_updates']['inserts_per_second']:.0f}/s)"
    )
    print(
        "incremental 1k retractions: "
        f"{result['incremental_updates']['retract_ms']:.2f} ms "
        f"({result['incremental_updates']['retractions_per_second']:.0f}/s)"
    )
    print(
        "max RSS: "
        f"{result['process_max_rss_kb']} KB"
    )
    print(f"manifest -> {out}")


if __name__ == "__main__":
    main()
