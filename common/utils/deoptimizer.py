"""
UOF (Universal OOD Features) — statement-level deoptimization transforms for SLODA-style attacks.

Uses Tree-sitter for span discovery and UTF-8 byte slicing for edits (AST is read-only).
"""
from __future__ import annotations

import re
from typing import Callable

from common.utils.parser import initialize_language_parser, src2tree


def _line_starts_at(node) -> int:
    """1-based line number where *node* starts."""
    return int(node.start_point[0]) + 1


def _line_byte_span(code: str, line_no: int) -> tuple[int, int, str] | None:
    """
    Return (start_byte, end_byte, segment) for line `line_no` (1-based).
    `segment` is the full line chunk as returned by splitlines(keepends=True).
    """
    lines = code.splitlines(keepends=True)
    if line_no < 1 or line_no > len(lines):
        return None
    start = 0
    for i in range(line_no - 1):
        start += len(lines[i].encode("utf-8"))
    seg = lines[line_no - 1]
    end = start + len(seg.encode("utf-8"))
    return start, end, seg


def _indent_of_line_segment(segment: str) -> str:
    body = segment.rstrip("\r\n")
    return body[: len(body) - len(body.lstrip())]


def _replace_bytes(code: str, start_byte: int, end_byte: int, new_text: str) -> str:
    b = code.encode("utf-8")
    return (b[:start_byte] + new_text.encode("utf-8") + b[end_byte:]).decode("utf-8")


def _is_plain_int_literal(node, source_bytes: bytes) -> bool:
    if node.type != "number_literal":
        return False
    text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    if "." in text or "e" in text.lower():
        return False
    if text.startswith(("0x", "0X")):
        return False
    try:
        int(text, 10)
    except ValueError:
        return False
    return True


def _split_int_value(n: int) -> tuple[int, int]:
    a = n // 2
    b = n - a
    return a, b


def _parenthesized_inner_span(node, source_bytes: bytes) -> tuple[int, int] | None:
    """For a parenthesized_expression, return byte span of the inner expression (no parens)."""
    if node.type != "parenthesized_expression":
        return None
    inner = None
    for ch in node.children:
        if ch.type not in ("(", ")"):
            inner = ch
            break
    if inner is None:
        return None
    return inner.start_byte, inner.end_byte


class CodeDeoptimizer:
    """Apply UOF deoptimization strategies at a single target line."""

    def apply_cf(self, code: str, target_line: int) -> str | None:
        """Constant folding reverse: `= n` -> `= a + b` with a+b=n."""
        source_bytes = code.encode("utf-8")
        parser = initialize_language_parser("cpp")
        root = src2tree(parser, source_bytes)

        def try_replace_value(value_node) -> str | None:
            if not _is_plain_int_literal(value_node, source_bytes):
                return None
            text = source_bytes[value_node.start_byte : value_node.end_byte].decode("utf-8")
            n = int(text, 10)
            a, b = _split_int_value(n)
            new_lit = f"{a} + {b}"
            return _replace_bytes(code, value_node.start_byte, value_node.end_byte, new_lit)

        def visit(node) -> str | None:
            if node.type == "assignment_expression" and _line_starts_at(node) == target_line:
                rhs = node.child_by_field_name("right")
                if rhs is not None and _is_plain_int_literal(rhs, source_bytes):
                    return try_replace_value(rhs)
            if node.type == "init_declarator" and _line_starts_at(node) == target_line:
                val = node.child_by_field_name("value")
                if val is None:
                    val = node.child_by_field_name("initializer")
                if val is not None and _is_plain_int_literal(val, source_bytes):
                    return try_replace_value(val)
            for c in node.children:
                r = visit(c)
                if r is not None:
                    return r
            return None

        return visit(root)

    def apply_ccp(self, code: str, target_line: int) -> str | None:
        """Wrap one line in a tautological if with two dummy ints."""
        span = _line_byte_span(code, target_line)
        if span is None:
            return None
        start_b, end_b, seg = span
        stripped = seg.rstrip("\r\n")
        if not stripped.strip():
            return None
        ws = _indent_of_line_segment(seg)
        body = stripped[len(ws) :]
        a, b = f"_uof_ccp_a_{target_line}", f"_uof_ccp_b_{target_line}"
        block = (
            f"{ws}int {a} = 2; int {b} = 3; if ({a} != {b}) {{\n"
            f"{ws}    {body}\n"
            f"{ws}}}\n"
        )
        return _replace_bytes(code, start_b, end_b, block)

    def apply_ubp(self, code: str, target_line: int) -> str | None:
        """Wrap one line in a once-executed tautological while (strlen)."""
        span = _line_byte_span(code, target_line)
        if span is None:
            return None
        start_b, end_b, seg = span
        stripped = seg.rstrip("\r\n")
        if not stripped.strip():
            return None
        ws = _indent_of_line_segment(seg)
        body = stripped[len(ws) :].rstrip(";")
        # keep statement inside loop; add semicolon if original had trailing stmt
        inner = f"{body};" if not body.endswith(";") else body
        block = (
            f"{ws}while (strlen(\"Cons\") > 2) {{\n"
            f"{ws}    {inner}\n"
            f"{ws}    break;\n"
            f"{ws}}}\n"
        )
        return _replace_bytes(code, start_b, end_b, block)

    def apply_be(self, code: str, target_line: int) -> str | None:
        """Extract if-condition into a bool temp before the if."""
        source_bytes = code.encode("utf-8")
        parser = initialize_language_parser("cpp")
        root = src2tree(parser, source_bytes)

        def visit(node) -> str | None:
            if node.type != "if_statement":
                for c in node.children:
                    r = visit(c)
                    if r is not None:
                        return r
                return None

            if _line_starts_at(node) != target_line:
                for c in node.children:
                    r = visit(c)
                    if r is not None:
                        return r
                return None

            cond = node.child_by_field_name("condition")
            if cond is None:
                for ch in node.children:
                    if ch.type == "parenthesized_expression":
                        cond = ch
                        break
            if cond is None:
                return None

            inner_span = _parenthesized_inner_span(cond, source_bytes)
            if inner_span is None:
                inner_start, inner_end = cond.start_byte, cond.end_byte
            else:
                inner_start, inner_end = inner_span

            cond_text = source_bytes[inner_start:inner_end].decode("utf-8").strip()
            if not cond_text:
                return None

            var = f"_uof_be_{target_line}"
            if_kw_start = node.start_byte
            prefix = source_bytes[:if_kw_start].decode("utf-8", errors="replace")
            last_line = prefix.split("\n")[-1] if prefix else ""
            indent = last_line[: len(last_line) - len(last_line.lstrip())]

            insert_b = f"{indent}bool {var} = {cond_text};\n".encode("utf-8")
            return (
                source_bytes[:if_kw_start]
                + insert_b
                + source_bytes[if_kw_start:cond.start_byte]
                + f"({var})".encode("utf-8")
                + source_bytes[cond.end_byte:]
            ).decode("utf-8")

        return visit(root)

    def _apply_osr_line_regex(self, code: str, target_line: int) -> str | None:
        """Fallback: rewrite shift-by-1 on a single line when binary_expression fields differ."""
        span = _line_byte_span(code, target_line)
        if span is None:
            return None
        start_b, end_b, seg = span
        line = seg.rstrip("\r\n")
        new_line = re.sub(r"<<\s*1\b", "* 2", line, count=1)
        new_line = re.sub(r">>\s*1\b", "/ 2", new_line, count=1)
        new_line = new_line.replace("<<1", "* 2").replace(">>1", "/ 2")
        if new_line == line:
            return None
        return _replace_bytes(code, start_b, end_b, new_line + seg[len(line) :])

    def apply_osr(self, code: str, target_line: int) -> str | None:
        """Replace `<< 1` / `>> 1` with `* 2` / `/ 2` on the target line (first match)."""
        source_bytes = code.encode("utf-8")
        parser = initialize_language_parser("cpp")
        root = src2tree(parser, source_bytes)

        def visit(node) -> str | None:
            if node.type == "binary_expression" and _line_starts_at(node) == target_line:
                right = node.child_by_field_name("right")
                left = node.child_by_field_name("left")
                op = node.child_by_field_name("operator")
                if right is not None and left is not None:
                    rtxt = source_bytes[right.start_byte : right.end_byte].decode("utf-8").strip()
                    optxt = ""
                    if op is not None:
                        optxt = source_bytes[op.start_byte : op.end_byte].decode("utf-8")
                    else:
                        mid = source_bytes[left.end_byte : right.start_byte].decode("utf-8")
                        optxt = mid.strip()
                    if rtxt == "1" and optxt in ("<<", ">>"):
                        seg = source_bytes[node.start_byte : node.end_byte].decode("utf-8")
                        new_seg = re.sub(r"<<\s*1\b", "* 2", seg, count=1)
                        new_seg = re.sub(r">>\s*1\b", "/ 2", new_seg, count=1)
                        if new_seg != seg:
                            return _replace_bytes(code, node.start_byte, node.end_byte, new_seg)
                        new_seg = seg.replace("<<1", "* 2").replace(">>1", "/ 2")
                        if new_seg != seg:
                            return _replace_bytes(code, node.start_byte, node.end_byte, new_seg)

            for c in node.children:
                r = visit(c)
                if r is not None:
                    return r
            return None

        r = visit(root)
        if r is not None:
            return r
        return self._apply_osr_line_regex(code, target_line)


def apply_uof(code: str, target_line: int) -> dict[str, str]:
    """
    Run all UOF strategies on `code`, each attempting a single edit anchored at `target_line`.

    Returns mapping strategy name -> full transformed source. Omitted keys mean no rewrite applied.
    """
    d = CodeDeoptimizer()
    strategies: list[tuple[str, Callable[[str, int], str | None]]] = [
        ("CF", d.apply_cf),
        ("CCP", d.apply_ccp),
        ("UBP", d.apply_ubp),
        ("BE", d.apply_be),
        ("OSR", d.apply_osr),
    ]
    out: dict[str, str] = {}
    for name, fn in strategies:
        try:
            res = fn(code, target_line)
        except Exception:
            res = None
        if res is not None and res != code:
            out[name] = res
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    sample = """#include <string.h>
int main(void) {
    int a = 5;
    int x = 10;
    if (x > 3) {
        x = 1;
    }
    int y = 4 << 1;
    int z = 8 >> 1;
    return 0;
}
"""
    # Line numbers 1-based in `sample`:
    tests = [
        (3, "CF on `int a = 5;`"),
        (4, "CF on `int x = 10;`"),
        (5, "BE on `if (x > 3)`"),
        (6, "CCP / UBP on `x = 1;`"),
        (8, "OSR on `<< 1`"),
        (9, "OSR on `>> 1`"),
    ]
    for line, note in tests:
        print("=" * 60)
        print(f"target_line={line} ({note})")
        results = apply_uof(sample, line)
        for k, v in results.items():
            print(f"--- [{k}] ---")
            print(v)
        if not results:
            print("(no strategy produced a change)")
