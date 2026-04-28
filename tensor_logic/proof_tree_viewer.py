from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ProofTreeNode:
    node_id: str
    head: tuple[str, str, str]
    children: tuple["ProofTreeNode", ...] = ()
    confidence: float | None = None
    reason: str | None = None
    source: tuple[str, int] | None = None
    is_negative: bool = False


def build_proof_tree_view(payload: dict) -> ProofTreeNode:
    if payload.get("answer") is True:
        proof = payload.get("proof")
        if not isinstance(proof, dict):
            raise ValueError("expected 'proof' object when answer=true")
        return _parse_positive_node(proof, "0")
    if payload.get("answer") is False:
        explanation = payload.get("explanation")
        if not isinstance(explanation, dict):
            raise ValueError("expected 'explanation' object when answer=false")
        return _parse_negative_node(explanation, "0")
    raise ValueError("payload must include boolean 'answer'")


def render_proof_tree(node: ProofTreeNode, collapsed: Iterable[str] | None = None) -> str:
    collapsed_set = set(collapsed or ())
    lines: list[str] = []
    _render(node, collapsed_set, lines, depth=0)
    return "\n".join(lines)


def _parse_positive_node(data: dict, node_id: str) -> ProofTreeNode:
    source = None
    src_obj = data.get("source")
    if isinstance(src_obj, dict):
        file_name = src_obj.get("file")
        lineno = src_obj.get("lineno")
        if isinstance(file_name, str) and isinstance(lineno, int):
            source = (file_name, lineno)
    confidence = data.get("confidence")
    return ProofTreeNode(
        node_id=node_id,
        head=_parse_head(data),
        children=tuple(_parse_positive_node(child, f"{node_id}.{i}") for i, child in enumerate(data.get("body", []))),
        confidence=float(confidence) if isinstance(confidence, (float, int)) else None,
        source=source,
    )


def _parse_negative_node(data: dict, node_id: str) -> ProofTreeNode:
    children = []
    for i, child in enumerate(data.get("body", [])):
        if isinstance(child, dict) and isinstance(child.get("explanation"), dict):
            children.append(_parse_negative_node(child["explanation"], f"{node_id}.{i}"))
        elif isinstance(child, dict):
            children.append(_parse_negative_node(child, f"{node_id}.{i}"))
    reason = data.get("reason")
    return ProofTreeNode(
        node_id=node_id,
        head=_parse_head(data),
        children=tuple(children),
        reason=reason if isinstance(reason, str) else None,
        is_negative=True,
    )


def _parse_head(data: dict) -> tuple[str, str, str]:
    raw_head = data.get("head")
    if not isinstance(raw_head, (list, tuple)) or len(raw_head) != 3:
        raise ValueError("expected head as [relation, src, dst]")
    rel, src, dst = raw_head
    return str(rel), str(src), str(dst)


def _render(node: ProofTreeNode, collapsed: set[str], out: list[str], depth: int) -> None:
    indent = "  " * depth
    has_children = bool(node.children)
    marker = "[+]" if node.node_id in collapsed and has_children else ("[-]" if has_children else "[*]")
    rel, src, dst = node.head
    text = f"{rel}({src}, {dst})"
    if node.is_negative:
        text += " = False"
    if node.reason:
        text += f" [{node.reason}]"
    if node.confidence is not None:
        text += f" ({node.confidence:.2f})"
    if node.source is not None:
        file_name, lineno = node.source
        text += f" [{file_name}:{lineno}]"
    out.append(f"{indent}{marker} {text}")
    if node.node_id in collapsed:
        return
    for child in node.children:
        _render(child, collapsed, out, depth + 1)
