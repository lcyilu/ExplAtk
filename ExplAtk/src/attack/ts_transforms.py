"""
ts_transforms.py — tree-sitter-based semantics-preserving code transformation module
============================================================

Intended to replace the regex-based data-flow / control-flow attack stages in expl_atk.py.
All transformations use *source-code line lists* as input and output, and internally use the tree-sitter AST
to precisely locate syntactic structures, avoiding regex issues such as parenthesis matching / type inference.

Design principles
--------
1. Each transform function uses a unified signature:
       transform_xxx(lines, target_line, root, **ctx)
           → (new_lines, success: bool, delta: int)
   where delta is the line-count change introduced by the transform (positive = inserted, negative = deleted).
2. Line-number convention: externally passed / returned line_no values are **1-indexed**;
   tree-sitter start_point/end_point.row values are **0-indexed**.
3. Type inference uses AST declaration search plus fallback heuristics and does not require full compilation.
4. All transformations preserve semantics (or insert redundant code indistinguishable in output).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

# ────────────────────────────────────────────────────────────
# tree-sitter imports (reuse the user's existing parser initialization setup)
# ────────────────────────────────────────────────────────────
from tree_sitter import Node

# User-provided utility functions
from common.ast_parser.run_parser import parse_code_to_ast  # type: ignore

# ────────────────────────────────────────────────────────────
# C keywords / reserved words (imported from the project's centralized keyword module)
# ────────────────────────────────────────────────────────────
from common.utils.keywords import (
    __builtin__funcs__,
    __key_words__,
    __macros__,
    __other__keywords__,
    __special_ids__,
)

_C_RESERVED: frozenset = frozenset().union(
    __key_words__,
    __macros__,
    __special_ids__,
    __builtin__funcs__,
    __other__keywords__,
)


# ══════════════════════════════════════════════════════════════
#  Part 0 ─ Constants and configuration
# ══════════════════════════════════════════════════════════════

# No global constants are currently used; keep Part 0 as a placeholder for future OOV-name allowlists, etc.


# ══════════════════════════════════════════════════════════════
#  Part 1 ─ Robust line-number mapping tracker
# ══════════════════════════════════════════════════════════════

class RobustLineTracker:
    """
    Maintain a bidirectional original_line → current_line mapping.

    Core idea:
      - Internally maintain a dict: orig → current (1-indexed)
      - resolve(orig) returns the current line number after all applicable offsets
      - After each transformation, call record_change(current_start, old_count, new_count)
        to update offsets

    Improvements over the original LineTracker:
      1. Supports interval replacement (old_span != new_span), not only insertion
      2. Tracks based on current line numbers instead of original line numbers to avoid offset crossings after multiple transforms
      3. Maintains an extra reverse mapping to support "current line → nearest original line" queries
    """

    def __init__(self, total_lines: int):
        self._fwd: Dict[int, int] = {i: i for i in range(1, total_lines + 1)}

    def resolve(self, original_line: int) -> int:
        """Original line number → current line number."""
        return self._fwd.get(original_line, original_line)

    def record_change(self, current_start: int, old_count: int, new_count: int):
        """
        Record one interval transformation:
            current lines [current_start, current_start + old_count)
            are replaced by new_count lines.
        All original lines mapped to >= current_start + old_count must be shifted by delta.
        """
        delta = new_count - old_count
        if delta == 0:
            return
        threshold = current_start + old_count
        for orig in self._fwd:
            if self._fwd[orig] >= threshold:
                self._fwd[orig] += delta

    def update_total(self, new_total: int):
        """Extend the mapping table when the transformed code changes total line count; newly added lines have no original-line counterpart."""
        pass

    def snapshot(self) -> Dict[int, int]:
        """Return a snapshot copy of the current mapping for debugging."""
        return dict(self._fwd)


# ══════════════════════════════════════════════════════════════
#  Part 2 ─ tree-sitter AST helper functions
# ══════════════════════════════════════════════════════════════

def _node_text(node: Node) -> str:
    """Safely get node text."""
    return node.text.decode("utf-8") if node and node.text else ""


def _get_indent(line: str) -> str:
    """Get the indentation prefix of a line."""
    m = re.match(r'^(\s*)', line)
    return m.group(1) if m else ""


def _find_nodes_by_type(root: Node, type_name: str) -> List[Node]:
    """Run DFS to find all nodes of the specified types."""
    results = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == type_name:
            results.append(node)
        stack.extend(reversed(node.children))
    return results


def _find_node_at_line(root: Node, line_0: int, type_name: str = None) -> Optional[Node]:
    """
    Find nodes whose start line (0-indexed) == line_0.
    If type_name is specified, match only that type.
    Return the outermost matching node.
    """
    best = None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_point[0] == line_0:
            if type_name is None or node.type == type_name:
                if best is None or (node.end_point[0] - node.start_point[0]) > (
                    best.end_point[0] - best.start_point[0]
                ):
                    best = node
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return best


def _find_statement_node_at_line(root: Node, line_0: int) -> Optional[Node]:
    """Find the statement-level node whose start line == line_0."""
    stmt_types = {
        "expression_statement", "declaration", "return_statement",
        "if_statement", "for_statement", "while_statement", "do_statement",
        "switch_statement", "compound_statement", "goto_statement",
        "break_statement", "continue_statement", "labeled_statement",
    }
    best = None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_point[0] == line_0 and node.type in stmt_types:
            if best is None or node.type in ("if_statement", "for_statement",
                                              "while_statement", "do_statement"):
                best = node
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return best


def _find_enclosing_node(root: Node, line_0: int, type_names: set) -> Optional[Node]:
    """Find the innermost node of the specified type that contains line_0."""
    best = None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in type_names and node.start_point[0] <= line_0 <= node.end_point[0]:
            if best is None or (node.end_point[0] - node.start_point[0]) < (
                best.end_point[0] - best.start_point[0]
            ):
                best = node
        for child in node.children:
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return best


def _find_enclosing_function(root: Node, line_0: int) -> Optional[Node]:
    """Find the function-definition node that contains line_0."""
    return _find_enclosing_node(root, line_0, {"function_definition"})


def _find_enclosing_loop(root: Node, line_0: int) -> Optional[Node]:
    """Find the innermost loop node that contains line_0."""
    return _find_enclosing_node(root, line_0, {"for_statement", "while_statement", "do_statement"})


def _get_compound_body_range(node: Node) -> Optional[Tuple[int, int]]:
    """
    Get the line range of a compound_statement (start_row_0, end_row_0).
    If body is not a compound_statement (single statement), return that statement's line range.
    """
    body = node.child_by_field_name("body")
    if body is None:
        body = node.child_by_field_name("consequence")
    if body is None:
        return None
    return (body.start_point[0], body.end_point[0])


def _unwrap_parenthesized(node: Node) -> Node:
    """Remove the parenthesized_expression wrapper and return the inner expression."""
    while node and node.type == "parenthesized_expression" and node.named_child_count > 0:
        node = node.named_children[0]
    return node


def _get_condition_node(stmt_node: Node) -> Optional[Node]:
    """
    Extract the condition-expression node from an if_statement / while_statement / do_statement.
    Automatically unwrap parenthesized_expression.
    """
    cond = stmt_node.child_by_field_name("condition")
    if cond is None:
        return None
    return _unwrap_parenthesized(cond)


# ── Part 2b ─ Safety-check functions ──────────────────────────────────

def _has_ast_errors(root: Node) -> bool:
    """Check whether the AST contains ERROR nodes."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            return True
        stack.extend(node.children)
    return False

def _has_error_near_line(root: Node, line_0: int, radius: int = 2) -> bool:
    """
    Check whether there are ERROR nodes near the target line (±radius lines).
    Skip only transformations near ERROR nodes without affecting distant lines.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            err_start = node.start_point[0]
            err_end = node.end_point[0]
            if err_start - radius <= line_0 <= err_end + radius:
                return True
        stack.extend(node.children)
    return False


def _is_inside_switch_case(root: Node, line_0: int) -> bool:
    """Check whether the target line is inside a switch-case structure."""
    switch_node = _find_enclosing_node(root, line_0, {"switch_statement"})
    if switch_node is None:
        return False
    body = switch_node.child_by_field_name("body")
    if body and body.start_point[0] < line_0 <= body.end_point[0]:
        return True
    return False


def _is_case_label_line(root: Node, line_0: int) -> bool:
    """Check whether the target line is a case/default label line."""
    node = _find_node_at_line(root, line_0, "case_statement")
    if node is not None:
        return True
    node = _find_node_at_line(root, line_0, "labeled_statement")
    return node is not None


def _is_macro_line(line: str) -> bool:
    """Check whether a line is a macro invocation or preprocessor directive."""
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if re.match(r'^[A-Z_][A-Z0-9_]*\s*\(', stripped):
        return True
    if re.match(r'^[A-Z_][A-Z0-9_]+\s*[({;]', stripped):
        return True
    return False


def _is_for_init_or_update(root: Node, line_0: int) -> bool:
    """Check whether the target line is in a for-loop header (the line containing init/cond/update)."""
    for_node = _find_enclosing_node(root, line_0, {"for_statement"})
    if for_node is None:
        return False
    if for_node.start_point[0] == line_0:
        return True
    return False


def _is_safe_for_insertion(root: Node, lines: List[str], line_0: int) -> bool:
    """
    Comprehensive safety check: whether a new line can be safely inserted before this line or the line can be replaced by multiple lines.

    Scenarios where insertion is forbidden:
      - Macro invocation lines
      - case/default label lines
      - Lines containing the init/update part of a for loop
    """
    if line_0 < 0 or line_0 >= len(lines):
        return False
    if _is_macro_line(lines[line_0]):
        return False
    if _is_case_label_line(root, line_0):
        return False
    if _is_for_init_or_update(root, line_0):
        return False
    return True


def _is_safe_for_block_transform(root: Node, lines: List[str], line_0: int) -> bool:
    """
    Check whether block-level transformations can be safely applied (if wrapping, condition splitting, etc.).
    More restrictive than _is_safe_for_insertion: additionally forbids inside switch-case structures.
    """
    if not _is_safe_for_insertion(root, lines, line_0):
        return False
    if _is_inside_switch_case(root, line_0):
        return False
    return True


# ══════════════════════════════════════════════════════════════
#  Part 3 ─ Type inference
# ══════════════════════════════════════════════════════════════

@dataclass
class VarTypeInfo:
    """Variable type information (best-effort inference)."""
    base_type: str = "int"
    is_pointer: bool = False
    is_unsigned: bool = False
    is_array: bool = False
    full_decl_type: str = ""
    confidence: float = 0.0


def _infer_type_from_ast(root: Node, var_name: str, use_line_0: int) -> VarTypeInfo:
    """
    Search the AST for the declaration of var_name and infer its type.

    Search strategy:
      1. Find all declaration nodes and check whether any declares var_name
      2. Search parameter declarations in function parameter lists
      3. Fall back to heuristics based on naming and usage patterns
    """
    info = VarTypeInfo()

    # ── Strategy 1: search declarations ──
    declarations = _find_nodes_by_type(root, "declaration")
    for decl in declarations:
        if decl.start_point[0] > use_line_0:
            continue
        decl_text = _node_text(decl)
        if not re.search(r'\b' + re.escape(var_name) + r'\b', decl_text):
            continue
        type_node = decl.child_by_field_name("type")
        if type_node:
            type_str = _node_text(type_node)
            info.base_type = type_str
            info.full_decl_type = type_str
            info.confidence = 0.8
            if '*' in decl_text.split(var_name)[0]:
                info.is_pointer = True
                info.full_decl_type += " *"
            if 'unsigned' in type_str:
                info.is_unsigned = True
            if '[' in decl_text:
                info.is_array = True
            return info

    # ── Strategy 2: search function parameters ──
    func_defs = _find_nodes_by_type(root, "function_definition")
    for func in func_defs:
        declarator = func.child_by_field_name("declarator")
        if declarator is None:
            continue
        params = _find_nodes_by_type(declarator, "parameter_declaration")
        for param in params:
            param_text = _node_text(param)
            if re.search(r'\b' + re.escape(var_name) + r'\b', param_text):
                type_node = param.child_by_field_name("type")
                if type_node:
                    type_str = _node_text(type_node)
                    info.base_type = type_str
                    info.full_decl_type = type_str
                    info.confidence = 0.7
                    if '*' in param_text:
                        info.is_pointer = True
                        info.full_decl_type += " *"
                    if 'unsigned' in type_str:
                        info.is_unsigned = True
                    return info

    # ── Strategy 3: heuristics ──
    if var_name in ('i', 'j', 'k', 'n', 'idx', 'index', 'count', 'len', 'size'):
        info.base_type = "int"
        info.full_decl_type = "int"
        info.confidence = 0.4
    elif var_name.startswith('p') or var_name.endswith('ptr'):
        info.is_pointer = True
        info.base_type = "void"
        info.full_decl_type = "void *"
        info.confidence = 0.2
    else:
        info.base_type = "int"
        info.full_decl_type = "int"
        info.confidence = 0.1

    return info


# ══════════════════════════════════════════════════════════════
#  Part 4 ─ W2V-aware name generator
# ══════════════════════════════════════════════════════════════

class NameGenerator:
    """
    Variable-name generator based on the W2V vocabulary.

    Three-level fallback strategy:
      1. Seed similarity retrieval: use the transformed variable as the seed and retrieve
         the most similar legal C identifiers from the W2V vocabulary.
      2. Generic-pool sampling: filter all legal C identifiers from the W2V vocabulary, keep the mid-frequency
         region, and randomly shuffle it as candidates.
      3. Counter fallback: v1, v2, ... as a last resort.

    Usage:
        gen = NameGenerator(wv, existing_ids)
        name = gen.generate_one(seed_var="level")
    """

    def __init__(self, wv, existing_ids: Set[str]):
        """
        Args:
            wv:           gensim KeyedVectors / Word2Vec vocabulary object
                          Must support wv.key_to_index and wv.most_similar()
            existing_ids: all identifiers already present in the current code
        """
        self.wv = wv
        self.existing_ids = existing_ids
        self._cache: Dict[str, List[str]] = {}
        self._generic_pool: Optional[List[str]] = None
        self._fallback_counter = 0
        # Optional MLM context for context-based naming candidates.
        # Injected by attack_structure_guided after constructing NameGenerator:
        #   ng.mlm_ctx = {"gen_candis_fn": fn, "mlm": mlm, "code_provider": lambda: cur_code}
        self.mlm_ctx: Optional[Dict[str, object]] = None

    def _in_vocab(self, word: str) -> bool:
        return word in self.wv.key_to_index

    def _is_valid_c_name(self, name: str) -> bool:
        """Check whether this is a legal and available C identifier."""
        if not name.isidentifier():
            return False
        if name in _C_RESERVED:
            return False
        if name in self.existing_ids:
            return False
        # Filter overly short names / names starting with digits (isidentifier has already excluded them)
        if len(name) < 2:
            return False
        return True

    def _split_compound(self, name: str) -> List[str]:
        """Split camelCase / snake_case into substring lists."""
        # snake_case
        parts = name.split('_')
        result = []
        for p in parts:
            # camelCase
            tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)', p)
            if tokens:
                result.extend(t.lower() for t in tokens)
            elif p:
                result.append(p.lower())
        return [r for r in result if len(r) >= 2]

    def _build_candidates_for_seed(self, seed: str, top_n: int = 150) -> List[str]:
        """Build a candidate list for the given seed (sorted, deduplicated, and filtered)."""
        raw_candidates = []

        if self._in_vocab(seed):
            try:
                sims = self.wv.most_similar(seed, topn=top_n)
                raw_candidates.extend(w for w, _ in sims)
            except (KeyError, ValueError):
                pass

        # If the seed is not in the vocabulary or there are too few candidates, try substring retrieval
        if len(raw_candidates) < 20:
            sub_parts = self._split_compound(seed)
            for sp in sub_parts:
                if self._in_vocab(sp):
                    try:
                        sims = self.wv.most_similar(sp, topn=50)
                        raw_candidates.extend(w for w, _ in sims)
                    except (KeyError, ValueError):
                        pass

        # Filter and deduplicate while preserving similarity order
        seen = set()
        filtered = []
        for c in raw_candidates:
            if c in seen:
                continue
            seen.add(c)
            if self._is_valid_c_name(c):
                filtered.append(c)
        return filtered

    def _ensure_generic_pool(self) -> None:
        """Lazily load the generic candidate pool."""
        if self._generic_pool is not None:
            return

        all_words = list(self.wv.key_to_index.keys())
        valid = [w for w in all_words if w.isidentifier() and w not in _C_RESERVED and len(w) >= 2]

        # Use the middle 60% frequency region, avoiding the most common and rarest tokens
        n = len(valid)
        start = int(n * 0.2)
        end = int(n * 0.8)
        pool = valid[start:end] if n > 10 else valid

        random.shuffle(pool)
        self._generic_pool = pool

    def generate_one(self, seed_var: str = "") -> str:
        """
        Generate a variable name that does not conflict with existing identifiers.

        Priority:
          1. MLM candidates (only when self.mlm_ctx has been injected)
             -- Based on the context of the current code + seed_var, select
                semantically reasonable identifiers within the word2vec vocabulary. The token stage has shown this path works better than pure W2V similarity.
          2. Seed similarity (W2V most_similar pool)
          3. Generic pool (legal identifiers with medium frequency in the W2V vocabulary)
          4. Counter fallback (v1, v2, ...)

        Args:
            seed_var: the original variable name being transformed, used for similarity/context retrieval.
                      If empty, skip strategies 1/2 and directly use the generic pool/counter.

        Returns:
            A legal C identifier in the W2V vocabulary.
        """
        # ── Strategy 1: MLM candidates (context-based) ──
        if seed_var and self.mlm_ctx is not None:
            gen_fn = self.mlm_ctx.get("gen_candis_fn")
            mlm = self.mlm_ctx.get("mlm")
            code_provider = self.mlm_ctx.get("code_provider")
            precomputed_provider = self.mlm_ctx.get("precomputed_provider")
            if gen_fn is not None and mlm is not None and callable(code_provider):
                try:
                    code_str = code_provider()
                    precomputed = (
                        precomputed_provider() if callable(precomputed_provider) else None
                    )
                    candis = gen_fn(code_str, mlm, seed_var, _precomputed=precomputed)
                except Exception:
                    candis = []
                for c in candis or []:
                    if not c or not isinstance(c, str):
                        continue
                    if not self._is_valid_c_name(c):
                        continue
                    self.existing_ids.add(c)
                    return c

        # ── Strategy 2: seed similarity ──
        if seed_var:
            if seed_var not in self._cache:
                self._cache[seed_var] = self._build_candidates_for_seed(seed_var)

            candidates = self._cache[seed_var]
            for c in candidates:
                if c not in self.existing_ids:
                    self.existing_ids.add(c)
                    return c

        # ── Strategy 3: generic pool ──
        self._ensure_generic_pool()
        for c in self._generic_pool:
            if c not in self.existing_ids and c not in _C_RESERVED:
                self.existing_ids.add(c)
                return c

        # ── Strategy 4: counter fallback ──
        while True:
            self._fallback_counter += 1
            name = f"v{self._fallback_counter}"
            if name not in self.existing_ids and name not in _C_RESERVED:
                self.existing_ids.add(name)
                return name

    def generate_batch(self, count: int, seed_var: str = "") -> List[str]:
        """Generate count unique variable names in batch."""
        return [self.generate_one(seed_var) for _ in range(count)]


def _collect_identifiers(root: Node) -> Set[str]:
    """Collect the text of all identifier and field_identifier nodes in the AST."""
    ids = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in ("identifier", "field_identifier"):
            ids.add(_node_text(node))
        stack.extend(node.children)
    return ids


# ══════════════════════════════════════════════════════════════
#  Part 5 ─ Data-flow semantics-preserving transformations
# ══════════════════════════════════════════════════════════════

def transform_temp_variable_insert(
    lines: List[str], target_line_1: int, var_name: str,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Temporary-variable insertion (type-aware version).
    Insert before target_line:
        <type> <tmp_name> = <var>;
        <var> = <tmp_name>;
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    type_info = _infer_type_from_ast(root, var_name, idx)
    if type_info.is_array:
        return lines, False, 0

    indent = _get_indent(lines[idx])

    if name_gen is not None:
        tmp_name = name_gen.generate_one(seed_var=var_name)
    else:
        tmp_name = f"_tmp_{random.randint(1000, 9999)}"
        existing_ids.add(tmp_name)

    type_str = type_info.full_decl_type or "int"
    insert_lines = [
        f"{indent}{type_str} {tmp_name} = {var_name};",
        f"{indent}{var_name} = {tmp_name};",
    ]

    result = list(lines)
    result[idx:idx] = insert_lines
    return result, True, 2


def transform_expression_decomposition(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Expression decomposition: split a compound expression into multi-step intermediate-variable assignments.

    Recognized pattern: var = A op B;  →  type _t1 = A; type _t2 = B; var = _t1 op _t2;

    Use tree-sitter to precisely locate binary_expression and avoid incorrectly splitting function calls.
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None:
        return lines, False, 0

    assign_node = None
    binary_node = None

    if stmt.type == "declaration":
        for child in _find_nodes_by_type(stmt, "init_declarator"):
            value = child.child_by_field_name("value")
            if value and value.type == "binary_expression":
                assign_node = stmt
                binary_node = value
                break
    elif stmt.type == "expression_statement":
        for child in _find_nodes_by_type(stmt, "assignment_expression"):
            right = child.child_by_field_name("right")
            if right and right.type == "binary_expression":
                assign_node = stmt
                binary_node = right
                break

    if binary_node is None:
        return lines, False, 0

    left = binary_node.child_by_field_name("left")
    right = binary_node.child_by_field_name("right")
    if left is None or right is None:
        return lines, False, 0

    op_text = None
    for child in binary_node.children:
        if child == left or child == right:
            continue
        if not child.is_named:
            op_text = _node_text(child).strip()
            break
    if op_text is None:
        return lines, False, 0

    left_text = _node_text(left).strip()
    right_text = _node_text(right).strip()

    type_str = "int"
    if stmt.type == "declaration":
        type_node = stmt.child_by_field_name("type")
        if type_node:
            type_str = _node_text(type_node)

    if name_gen is not None:
        # Use identifiers in the left/right operands as the seed
        seed = ""
        if left.type == "identifier":
            seed = left_text
        elif right.type == "identifier":
            seed = right_text
        tmp1 = name_gen.generate_one(seed_var=seed)
        tmp2 = name_gen.generate_one(seed_var=seed)
    else:
        tmp1 = f"_sub_{random.randint(1000, 9999)}"
        tmp2 = f"_sub_{random.randint(1000, 9999)}"
        existing_ids.update({tmp1, tmp2})

    bin_text = _node_text(binary_node)
    new_expr = f"{tmp1} {op_text} {tmp2}"
    new_stmt_line = line.replace(bin_text, new_expr, 1)

    insert_lines = [
        f"{indent}{type_str} {tmp1} = {left_text};",
        f"{indent}{type_str} {tmp2} = {right_text};",
        new_stmt_line,
    ]

    result = list(lines)
    result[idx:idx + 1] = insert_lines
    return result, True, 2


def transform_propagation_chain(
    lines: List[str], src_line_1: int, dst_line_1: int, var_name: str,
    root: Node, existing_ids: Set[str], chain_length: int = 2,
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Variable propagation-chain extension: insert an equivalent assignment chain between variable definition and use.

        int x = compute();      int x = compute();
                            →   int _r1 = x;
        use(x);                 int _r2 = _r1;
                                use(_r2);
    """
    src_idx = src_line_1 - 1
    dst_idx = dst_line_1 - 1
    if src_idx < 0 or dst_idx < 0 or src_idx >= len(lines) or dst_idx >= len(lines):
        return lines, False, 0
    if dst_idx <= src_idx:
        return lines, False, 0

    if not _is_safe_for_insertion(root, lines, dst_idx):
        return lines, False, 0

    type_info = _infer_type_from_ast(root, var_name, src_idx)
    if type_info.is_array:
        return lines, False, 0

    type_str = type_info.full_decl_type or "int"
    indent = _get_indent(lines[dst_idx])

    relay_names = []
    for _ in range(chain_length):
        if name_gen is not None:
            name = name_gen.generate_one(seed_var=var_name)
        else:
            name = f"_relay_{random.randint(1000, 9999)}"
            existing_ids.add(name)
        relay_names.append(name)

    insert_lines = []
    prev = var_name
    for rn in relay_names:
        insert_lines.append(f"{indent}{type_str} {rn} = {prev};")
        prev = rn

    final_name = relay_names[-1]
    pattern = r'\b' + re.escape(var_name) + r'\b'
    dst_line = lines[dst_idx]
    lhs_pat = r'^\s*' + re.escape(var_name) + r'\s*=[^=]'
    if re.match(lhs_pat, dst_line):
        return lines, False, 0
    new_dst_line = re.sub(pattern, final_name, dst_line)

    result = list(lines)
    result[dst_idx:dst_idx] = insert_lines
    result[dst_idx + chain_length] = new_dst_line

    return result, True, chain_length


def transform_assignment_split(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Assignment splitting (type-aware version; does not use __auto_type).

    var = expr;  →  <type> _split_tmp = expr; var = _split_tmp;
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None:
        return lines, False, 0

    indent = _get_indent(lines[idx])

    if stmt.type == "expression_statement":
        assigns = _find_nodes_by_type(stmt, "assignment_expression")
        if not assigns:
            return lines, False, 0
        assign = assigns[0]
        op_node = assign.child_by_field_name("operator")
        if op_node and _node_text(op_node).strip() != "=":
            return lines, False, 0

        left = assign.child_by_field_name("left")
        right = assign.child_by_field_name("right")
        if left is None or right is None:
            return lines, False, 0

        lhs_text = _node_text(left).strip()
        rhs_text = _node_text(right).strip()

        type_info = _infer_type_from_ast(root, lhs_text, idx)
        type_str = type_info.full_decl_type or "int"

        if name_gen is not None:
            tmp_name = name_gen.generate_one(seed_var=lhs_text)
        else:
            tmp_name = f"_split_{random.randint(1000, 9999)}"
            existing_ids.add(tmp_name)

        new_lines = [
            f"{indent}{type_str} {tmp_name} = {rhs_text};",
            f"{indent}{lhs_text} = {tmp_name};",
        ]
        result = list(lines)
        result[idx:idx + 1] = new_lines
        return result, True, 1

    elif stmt.type == "declaration":
        type_node = stmt.child_by_field_name("type")
        if type_node is None:
            return lines, False, 0

        init_decls = _find_nodes_by_type(stmt, "init_declarator")
        if not init_decls:
            return lines, False, 0

        init_decl = init_decls[0]
        value = init_decl.child_by_field_name("value")
        declarator = init_decl.child_by_field_name("declarator")
        if value is None or declarator is None:
            return lines, False, 0

        type_str = _node_text(type_node).strip()
        var_text = _node_text(declarator).strip()
        val_text = _node_text(value).strip()

        ptr_prefix = ""
        if var_text.startswith("*"):
            ptr_prefix = "*"
            var_text = var_text.lstrip("*").strip()
            type_str = type_str + " *"

        if name_gen is not None:
            tmp_name = name_gen.generate_one(seed_var=var_text)
        else:
            tmp_name = f"_split_{random.randint(1000, 9999)}"
            existing_ids.add(tmp_name)

        new_lines = [
            f"{indent}{type_str} {tmp_name} = {val_text};",
            f"{indent}{type_str} {ptr_prefix}{var_text} = {tmp_name};",
        ]
        result = list(lines)
        result[idx:idx + 1] = new_lines
        return result, True, 1

    return lines, False, 0


# ── Part 5b ─ Literal extraction / compound assignment / increment-decrement expansion (tree-sitter version) ──


def _find_number_literals_on_line(root: Node, line_0: int) -> List[Node]:
    """Find all number_literal nodes on the specified line."""
    results = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "number_literal" and node.start_point[0] == line_0:
            results.append(node)
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return results


def _find_string_literals_on_line(root: Node, line_0: int) -> List[Node]:
    """Find all string_literal nodes on the specified line."""
    results = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "string_literal" and node.start_point[0] == line_0:
            results.append(node)
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return results


def transform_constant_extraction(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Constant extraction: extract numeric / string literals into named temporary variables.

    Original:    header_nlines = 1 + image->ncolors;
    Transformed: int _c1 = 1;
            header_nlines = _c1 + image->ncolors;

    Original:    strcpy(s, " XPMEXT");
    Transformed: char _s1[] = " XPMEXT";
            strcpy(s, _s1);

    Impact on the PDG:
      - Each extraction introduces a new definition node and a new DDG edge
      - Changes the token features of the original node (numbers/strings disappear, variable names appear)
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    # ── Collect numeric literals on this line ──
    num_literals = _find_number_literals_on_line(root, idx)
    str_literals = _find_string_literals_on_line(root, idx)

    if not num_literals and not str_literals:
        return lines, False, 0

    # Replace from right to left by column position to avoid offsets
    all_literals = []
    for node in num_literals:
        text = _node_text(node)
        # Skip overly complex literals (such as floating-point scientific notation) and literals in #define
        if not text or line.strip().startswith("#"):
            continue
        # Skip simple indices in array subscripts (such as header[0]) but keep those in expressions
        all_literals.append(("number", node, text))

    for node in str_literals:
        text = _node_text(node)
        if not text or line.strip().startswith("#"):
            continue
        all_literals.append(("string", node, text))

    if not all_literals:
        return lines, False, 0

    # Sort from right to left by column position
    all_literals.sort(key=lambda x: x[1].start_point[1], reverse=True)

    insert_lines = []
    new_line = line
    extracted_any = False

    for lit_type, node, text in all_literals:
        # Compute start/end columns within the line
        col_start = node.start_point[1]
        col_end = node.end_point[1]

        if name_gen is not None:
            # Use a hint derived from the literal value as the seed
            seed = "num" if lit_type == "number" else "str"
            tmp_name = name_gen.generate_one(seed_var=seed)
        else:
            tmp_name = f"_const_{random.randint(1000, 9999)}"
            existing_ids.add(tmp_name)

        if lit_type == "number":
            # Infer type: integer vs floating point
            if '.' in text or 'e' in text.lower() or 'f' in text.lower():
                decl_type = "double"
            else:
                decl_type = "int"
            insert_lines.append(f"{indent}{decl_type} {tmp_name} = {text};")
        else:
            # String literal
            # char name[] = "..." is safer than char *name
            insert_lines.append(f"{indent}char {tmp_name}[] = {text};")

        # Replace in-line from right to left, offset-safe
        new_line = new_line[:col_start] + tmp_name + new_line[col_end:]
        extracted_any = True

    if not extracted_any:
        return lines, False, 0

    result = list(lines)
    # Insert declaration lines before the current line
    result[idx:idx + 1] = insert_lines + [new_line]
    delta = len(insert_lines)  # Added len(insert_lines) lines
    return result, True, delta


def _find_compound_assignment_on_line(root: Node, line_0: int) -> Optional[Node]:
    """
    Find the compound_assignment_expression or
    augmented_assignment_expression node on the specified line.
    In tree-sitter-c, +=, -=, *=, /=, etc. correspond to assignment_expression
    where operator is not "=".
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_point[0] == line_0 and node.type == "assignment_expression":
            # Check whether operator is a compound assignment
            for child in node.children:
                if not child.is_named:
                    op = _node_text(child).strip()
                    if op in ("+=", "-=", "*=", "/=", "%=", "<<=", ">>=",
                              "&=", "|=", "^="):
                        return node
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return None


def transform_compound_assignment_expand(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Compound-assignment expansion (relay-enhanced version, B.6):
        x += expr;
        →
        <type> _t = expr;
        x = x + _t;

    Impact on the PDG:
      - 1 node → 2 nodes (relay declaration + expanded assignment)
      - Changes the original node token set: original `{x, +=, expr_tokens}` becomes
        `{x, =, x, +, _t}` -- all RHS expression tokens are moved to the relay node
      - The DDG edge path is lengthened: variables in the original RHS take one extra hop to x

    Fallback: when name_gen is unavailable or the LHS type cannot be inferred, fall back to the old pure-expansion version
    (`x += expr → x = x + (expr);` only changes the original node tokens and does not add a node).
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    line = lines[idx]
    if _is_macro_line(line):
        return lines, False, 0

    assign_node = _find_compound_assignment_on_line(root, idx)
    if assign_node is None:
        return lines, False, 0

    left = assign_node.child_by_field_name("left")
    right = assign_node.child_by_field_name("right")
    if left is None or right is None:
        return lines, False, 0

    # Extract operator
    op_text = None
    for child in assign_node.children:
        if not child.is_named:
            op_text = _node_text(child).strip()
            if len(op_text) >= 2 and op_text.endswith("="):
                break

    if op_text is None or not op_text.endswith("="):
        return lines, False, 0

    # += → +, -= → -, etc.
    base_op = op_text[:-1]
    lhs_text = _node_text(left).strip()
    rhs_text = _node_text(right).strip()

    indent = _get_indent(line)

    # Get the full statement range
    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None:
        return lines, False, 0

    stmt_start = stmt.start_point[0]
    stmt_end = stmt.end_point[0]
    old_count = stmt_end - stmt_start + 1

    # ── Main path: relay-based expansion ──
    # Infer the LHS type as the relay variable declaration type; fall back to "int" if inference fails
    type_info = _infer_type_from_ast(root, lhs_text, idx)
    type_str = type_info.full_decl_type or "int"
    if "*" in type_str and not type_str.endswith("*"):
        # Forms like "int *" are OK; for complex forms like "int * const *", conservatively fall back to the old path
        pass

    if name_gen is not None:
        tmp_name = name_gen.generate_one(seed_var=lhs_text)
    else:
        tmp_name = f"_cmp_{random.randint(1000, 9999)}"
        existing_ids.add(tmp_name)

    new_lines_block = [
        f"{indent}{type_str} {tmp_name} = {rhs_text};",
        f"{indent}{lhs_text} = {lhs_text} {base_op} {tmp_name};",
    ]

    result = list(lines)
    result[stmt_start:stmt_end + 1] = new_lines_block
    delta = len(new_lines_block) - old_count
    return result, True, delta


def _find_update_expression_on_line(root: Node, line_0: int) -> Optional[Node]:
    """Find the update_expression (++/--) node on the specified line."""
    stack = [root]
    best = None
    while stack:
        node = stack.pop()
        if node.start_point[0] == line_0 and node.type == "update_expression":
            best = node
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return best


def transform_unary_increment_expand(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Increment/decrement operator expansion:
        x++;  →  int _c = 1; x = x + _c;
        x--;  →  int _c = 1; x = x - _c;

    Impact on the PDG:
      - Introduces a new constant-definition node and DDG edge
      - Splits compact ++ operations into multiple steps, increasing graph node count
      - Changes node token features
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    update_node = _find_update_expression_on_line(root, idx)
    if update_node is None:
        return lines, False, 0

    node_text = _node_text(update_node).strip()

    # Determine whether this is ++ or --, and whether it is prefix or postfix
    if node_text.endswith("++"):
        var_name = node_text[:-2].strip()
        op = "+"
    elif node_text.startswith("++"):
        var_name = node_text[2:].strip()
        op = "+"
    elif node_text.endswith("--"):
        var_name = node_text[:-2].strip()
        op = "-"
    elif node_text.startswith("--"):
        var_name = node_text[2:].strip()
        op = "-"
    else:
        return lines, False, 0

    if not var_name:
        return lines, False, 0

    # Check whether this update_expression is a standalone statement rather than a for-loop update clause or part of a larger expression
    # If it is a direct child of expression_statement, it can be safely expanded
    stmt = _find_statement_node_at_line(root, idx)

    if name_gen is not None:
        const_name = name_gen.generate_one(seed_var=var_name)
    else:
        const_name = f"_inc_{random.randint(1000, 9999)}"
        existing_ids.add(const_name)

    new_lines = [
        f"{indent}int {const_name} = 1;",
        f"{indent}{var_name} = {var_name} {op} {const_name};",
    ]

    # If it is a standalone statement, replace the whole line
    if stmt is not None and stmt.type == "expression_statement":
        stmt_start = stmt.start_point[0]
        stmt_end = stmt.end_point[0]
        old_count = stmt_end - stmt_start + 1
        result = list(lines)
        result[stmt_start:stmt_end + 1] = new_lines
        delta = len(new_lines) - old_count
        return result, True, delta
    else:
        # If embedded in a larger expression (such as a for update), only perform in-line replacement
        # x++ → (x = x + 1) -- but this is not very safe for for updates, so skip for now
        return lines, False, 0


# ── Part 5c ─ New substitutive structure transformations (A.5 / A.6 / A.7 / B.5) ──


def _find_first_subscript_on_line(root: Node, line_0: int) -> Optional[Node]:
    """Find the first subscript_expression (array access) node on the specified line."""
    stack = [root]
    while stack:
        node = stack.pop()
        if (node.start_point[0] == line_0
                and node.type == "subscript_expression"):
            return node
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return None


def transform_array_index_extract(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Array-index expression extraction (A.5):
        buf[i + j * stride]    →    int idx = i + j * stride;
                                    buf[idx]
    Trigger only for "non-trivial" indices (not a single identifier or a single literal) --
    `arr[i]` will not be rewritten, avoiding valueless idx relays.

    PDG impact:
      - The original subscript node token set is significantly simplified: index-expression tokens are removed,
        and one idx identifier is added
      - Adds a declaration node + DDG edge
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0
    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    sub = _find_first_subscript_on_line(root, idx)
    if sub is None:
        return lines, False, 0

    # subscript_expression: argument(array), index(subscript)
    index_node = sub.child_by_field_name("index")
    if index_node is None:
        return lines, False, 0

    # Skip simple indices: a single identifier / numeric literal
    if index_node.type in ("identifier", "number_literal"):
        return lines, False, 0

    index_text = _node_text(index_node).strip()
    if not index_text:
        return lines, False, 0

    # Choose the seed: prefer an identifier from index_text; otherwise use the array name
    arg_node = sub.child_by_field_name("argument")
    seed = ""
    if index_node.type == "binary_expression":
        for ch in index_node.children:
            if ch.type == "identifier":
                seed = _node_text(ch)
                break
    if not seed and arg_node is not None and arg_node.type == "identifier":
        seed = _node_text(arg_node)

    if name_gen is not None:
        tmp_name = name_gen.generate_one(seed_var=seed)
    else:
        tmp_name = f"_idx_{random.randint(1000, 9999)}"
        existing_ids.add(tmp_name)

    # In-line replacement: replace the original index with tmp_name using the column range
    col_start = index_node.start_point[1]
    col_end = index_node.end_point[1]
    new_line = line[:col_start] + tmp_name + line[col_end:]

    insert_lines = [f"{indent}int {tmp_name} = {index_text};"]
    result = list(lines)
    result[idx:idx + 1] = insert_lines + [new_line]
    return result, True, len(insert_lines)


def _find_first_call_on_line(root: Node, line_0: int) -> Optional[Node]:
    """Find the first call_expression node on the specified line."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_point[0] == line_0 and node.type == "call_expression":
            return node
        for child in reversed(node.children):
            if child.start_point[0] <= line_0 <= child.end_point[0]:
                stack.append(child)
    return None


def transform_call_arg_extract(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    Function-call argument extraction (A.6):
        memcpy(dst, src, len * 2)  →  int n = len * 2;
                                       memcpy(dst, src, n);

    Extract the last non-trivial argument of the call into a named relay; if all arguments are
    single identifiers / literals, skip it to avoid generating valueless candidates.

    PDG impact:
      - The call-node token set is significantly simplified (complex argument expressions are replaced by a single identifier)
      - Adds a declaration + DDG edge
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0
    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    call_node = _find_first_call_on_line(root, idx)
    if call_node is None:
        return lines, False, 0

    args_node = call_node.child_by_field_name("arguments")
    if args_node is None or args_node.named_child_count == 0:
        return lines, False, 0

    # Search backward for the first "complex argument"
    target_arg = None
    for arg in reversed(args_node.named_children):
        if arg.type in ("identifier", "number_literal", "string_literal",
                        "char_literal"):
            continue
        # Unwrap parenthesized_expression and inspect the inside
        inner = _unwrap_parenthesized(arg)
        if inner.type in ("identifier", "number_literal"):
            continue
        target_arg = arg
        break

    if target_arg is None:
        return lines, False, 0

    arg_text = _node_text(target_arg).strip()
    if not arg_text:
        return lines, False, 0

    # Choose seed: take the first identifier from arg_text; otherwise use the called function name
    seed = ""
    stack = [target_arg]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            seed = _node_text(n)
            break
        stack.extend(reversed(n.children))
    if not seed:
        fn_node = call_node.child_by_field_name("function")
        if fn_node is not None and fn_node.type == "identifier":
            seed = _node_text(fn_node)

    if name_gen is not None:
        tmp_name = name_gen.generate_one(seed_var=seed)
    else:
        tmp_name = f"_arg_{random.randint(1000, 9999)}"
        existing_ids.add(tmp_name)

    # In-line replacement: replace this argument with tmp_name using the column range
    col_start = target_arg.start_point[1]
    col_end = target_arg.end_point[1]
    # Replace only when the argument is fully within this idx line; multi-line arguments are too complex, so skip
    if (target_arg.start_point[0] != idx or target_arg.end_point[0] != idx):
        return lines, False, 0
    new_line = line[:col_start] + tmp_name + line[col_end:]

    insert_lines = [f"{indent}int {tmp_name} = {arg_text};"]
    result = list(lines)
    result[idx:idx + 1] = insert_lines + [new_line]
    return result, True, len(insert_lines)


def transform_return_extract(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    return-expression extraction (A.7):
        return a + b * c;   →   int ret = a + b * c;
                                return ret;

    Skip `return;` (no expression) and `return single_var;` (already a single identifier).

    PDG impact:
      - The return node token set changes from `{return, a, +, b, *, c}` to
        `{return, ret}` -- much lower information density
      - Adds a declaration node + DDG edge
      - return nodes are often marked as key nodes by the explainer (vulnerabilities are highly related to return values),
        so substitutive attacks targeting them have high ROI.
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0
    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None or stmt.type != "return_statement":
        return lines, False, 0

    # In the named children of return_statement, the first non-"return" keyword is the expression
    expr_node = None
    for ch in stmt.named_children:
        expr_node = ch
        break
    if expr_node is None:
        return lines, False, 0

    inner = _unwrap_parenthesized(expr_node)
    # Skip trivial returns
    if inner.type in ("identifier", "number_literal", "string_literal",
                      "char_literal"):
        return lines, False, 0

    expr_text = _node_text(expr_node).strip()
    if not expr_text:
        return lines, False, 0

    # Conservatively skip expressions spanning multiple lines because replacing the original line is inconvenient
    if (expr_node.start_point[0] != idx or expr_node.end_point[0] != idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    # Choose seed: take the first identifier from expr
    seed = ""
    stack = [expr_node]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            seed = _node_text(n)
            break
        stack.extend(reversed(n.children))

    if name_gen is not None:
        tmp_name = name_gen.generate_one(seed_var=seed or "ret")
    else:
        tmp_name = f"_ret_{random.randint(1000, 9999)}"
        existing_ids.add(tmp_name)

    # Infer the return type from the enclosing function declarator; fall back to "int" if inference fails
    type_str = "int"
    func = _find_enclosing_function(root, idx)
    if func is not None:
        type_node = func.child_by_field_name("type")
        if type_node is not None:
            type_str = _node_text(type_node).strip() or "int"

    new_lines_block = [
        f"{indent}{type_str} {tmp_name} = {expr_text};",
        f"{indent}return {tmp_name};",
    ]
    result = list(lines)
    result[idx:idx + 1] = new_lines_block
    return result, True, len(new_lines_block) - 1


def transform_multivar_decl_split(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
) -> Tuple[List[str], bool, int]:
    """
    Multi-variable declaration splitting (B.5):
        int a, b, c;          →    int a;
                                    int b;
                                    int c;
        int x = 1, y = 2;     →    int x = 1;
                                    int y = 2;

    PDG impact:
      - 1 node → N nodes (each declaration is independent)
      - Each new node token set is significantly simplified (no shared comma/type)
      - Does not rely on name_gen; purely structural splitting.
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0
    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None or stmt.type != "declaration":
        return lines, False, 0

    # Must contain at least 2 declarators (init_declarator or identifier-style declarator)
    declarators: List[Node] = []
    for ch in stmt.named_children:
        if ch.type in ("init_declarator", "identifier",
                       "pointer_declarator", "array_declarator"):
            declarators.append(ch)
    if len(declarators) < 2:
        return lines, False, 0

    type_node = stmt.child_by_field_name("type")
    if type_node is None:
        return lines, False, 0
    type_str = _node_text(type_node).strip()
    if not type_str:
        return lines, False, 0

    # Conservatively skip multi-line declarations
    if (stmt.start_point[0] != idx or stmt.end_point[0] != idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    new_lines_block: List[str] = []
    for decl in declarators:
        decl_text = _node_text(decl).strip()
        if not decl_text:
            continue
        new_lines_block.append(f"{indent}{type_str} {decl_text};")

    if len(new_lines_block) < 2:
        return lines, False, 0

    result = list(lines)
    result[idx:idx + 1] = new_lines_block
    return result, True, len(new_lines_block) - 1


# ══════════════════════════════════════════════════════════════
#  Part 6 ─ Control-flow semantics-preserving transformations
# ══════════════════════════════════════════════════════════════

# Current control-flow transformations:
#   - Ternary expression → if-else (B.4): see transform_ternary_to_if below
# Deprecated (weak perturbation under sum-pooled token embedding):
#   for_to_while / while_to_dowhile / while_to_for /
#   demorgan / split_and / split_or / dead_branch / early_return


def transform_ternary_to_if(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
) -> Tuple[List[str], bool, int]:
    """
    Ternary expression → if-else transformation (B.4).

    `var = (cond) ? a : b;`            →
    `if (cond) { var = a; } else { var = b; }`

    `<type> var = (cond) ? a : b;`     →
    `<type> var; if (cond) { var = a; } else { var = b; }`

    Impact on the PDG:
      - 1 node → 4 nodes (if / then / else / merge)
      - Edge types change from simple DDG to CDG + DDG
      - Perturbs both token sets and edge types, making it one of the few still-effective CDG-style transformations.
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    if not _is_safe_for_block_transform(root, lines, idx):
        return lines, False, 0

    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None:
        return lines, False, 0

    if stmt.type not in ("expression_statement", "declaration"):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    # Find the first conditional_expression (ternary) node
    cond_exprs = _find_nodes_by_type(stmt, "conditional_expression")
    if not cond_exprs:
        return lines, False, 0
    cond_expr = cond_exprs[0]

    # Three components of the ternary: condition, consequence, alternative
    condition = cond_expr.child_by_field_name("condition")
    consequence = cond_expr.child_by_field_name("consequence")
    alternative = cond_expr.child_by_field_name("alternative")
    if condition is None or consequence is None or alternative is None:
        return lines, False, 0

    cond_text = _node_text(condition).strip()
    cons_text = _node_text(consequence).strip()
    alt_text = _node_text(alternative).strip()

    # Distinguish two host statement types
    if stmt.type == "expression_statement":
        # Form: var = cond ? a : b;
        assigns = _find_nodes_by_type(stmt, "assignment_expression")
        if not assigns:
            return lines, False, 0
        assign = assigns[0]
        op_node = assign.child_by_field_name("operator")
        if op_node and _node_text(op_node).strip() != "=":
            return lines, False, 0
        left = assign.child_by_field_name("left")
        if left is None:
            return lines, False, 0
        lhs_text = _node_text(left).strip()

        new_lines = [
            f"{indent}if ({cond_text}) {{",
            f"{indent}    {lhs_text} = {cons_text};",
            f"{indent}}} else {{",
            f"{indent}    {lhs_text} = {alt_text};",
            f"{indent}}}",
        ]
        result = list(lines)
        result[idx:idx + 1] = new_lines
        return result, True, len(new_lines) - 1

    # stmt.type == "declaration": form int var = cond ? a : b;
    type_node = stmt.child_by_field_name("type")
    if type_node is None:
        return lines, False, 0
    type_str = _node_text(type_node).strip()

    init_decls = _find_nodes_by_type(stmt, "init_declarator")
    if not init_decls:
        return lines, False, 0
    declarator = init_decls[0].child_by_field_name("declarator")
    if declarator is None:
        return lines, False, 0
    var_text = _node_text(declarator).strip()

    new_lines = [
        f"{indent}{type_str} {var_text};",
        f"{indent}if ({cond_text}) {{",
        f"{indent}    {var_text} = {cons_text};",
        f"{indent}}} else {{",
        f"{indent}    {var_text} = {alt_text};",
        f"{indent}}}",
    ]
    result = list(lines)
    result[idx:idx + 1] = new_lines
    return result, True, len(new_lines) - 1



# ══════════════════════════════════════════════════════════════
#  Part 7 ─ Transformation registry and priority
# ══════════════════════════════════════════════════════════════

@dataclass
class TransformCandidate:
    """A transformation candidate to be evaluated."""
    name: str
    apply_fn: object
    kwargs: dict = field(default_factory=dict)
    priority: float = 0.0

    affects_ddg: bool = False
    affects_cdg: bool = False


def _build_ddg_transforms(
    lines: List[str], edge: dict, root: Node,
    existing_ids: Set[str], rename_map: dict,
    name_gen: Optional[NameGenerator] = None,
) -> List[TransformCandidate]:
    """
    Generate all applicable transformation candidates for one DDG edge.

    Design principles (cleaned version):
      All candidates satisfy the principle of "substantially changing a PDG node token set" --
      either the original-line RHS is extracted to a relay variable (A series), or literals/arguments/indices
      are rearranged into new named variables; transformations that only "add nodes without changing the original line"
    """
    candidates = []
    src_line = edge.get("src_line")
    dst_line = edge.get("dst_line")
    dep_var = edge.get("dep_variable", "")

    if dep_var:
        current = dep_var
        visited = set()
        while current in rename_map and current not in visited:
            visited.add(current)
            current = rename_map[current]
        dep_var = current

    if not dep_var or src_line is None or dst_line is None:
        return candidates

    importance = edge.get("importance", 0.5)

    # ── Temporary-variable insertion (observed effective on sample 1; keep the old logic) ──
    candidates.append(TransformCandidate(
        name="temp_var_insert",
        apply_fn=transform_temp_variable_insert,
        kwargs={"target_line_1": src_line, "var_name": dep_var,
                "root": root, "existing_ids": existing_ids,
                "name_gen": name_gen},
        priority=importance * 1.0,
        affects_ddg=True,
    ))

    # ── A.1 compound-expression splitting ──
    candidates.append(TransformCandidate(
        name="expr_decompose",
        apply_fn=transform_expression_decomposition,
        kwargs={"target_line_1": src_line,
                "root": root, "existing_ids": existing_ids,
                "name_gen": name_gen},
        priority=importance * 1.2,
        affects_ddg=True,
    ))

    # ── A.2 generic RHS relay (assignment_split; covers single-variable/call/array-access RHS) ──
    candidates.append(TransformCandidate(
        name="rhs_relay",
        apply_fn=transform_assignment_split,
        kwargs={"target_line_1": src_line,
                "root": root, "existing_ids": existing_ids,
                "name_gen": name_gen},
        priority=importance * 1.1,
        affects_ddg=True,
    ))

    # ── Propagation-chain extension (observed effective on sample 1; chain_length=1 reduces code bloat) ──
    if dst_line > src_line:
        candidates.append(TransformCandidate(
            name="propagation_chain",
            apply_fn=transform_propagation_chain,
            kwargs={"src_line_1": src_line, "dst_line_1": dst_line,
                    "var_name": dep_var,
                    "root": root, "existing_ids": existing_ids,
                    "chain_length": random.choice([1, 2]),
                    "name_gen": name_gen},
            priority=importance * 1.1,
            affects_ddg=True,
        ))

    # ── Literal extraction (numeric/string literals → named relay) ──
    for line_1 in {src_line, dst_line}:
        if line_1 is not None:
            candidates.append(TransformCandidate(
                name=f"const_extract_L{line_1}",
                apply_fn=transform_constant_extraction,
                kwargs={"target_line_1": line_1,
                        "root": root, "existing_ids": existing_ids,
                        "name_gen": name_gen},
                priority=importance * 1.4,
                affects_ddg=True,
            ))

    # ── B.6 compound-assignment expansion (upgraded version with RHS relay) ──
    for line_1 in {src_line, dst_line}:
        if line_1 is not None:
            candidates.append(TransformCandidate(
                name=f"compound_expand_L{line_1}",
                apply_fn=transform_compound_assignment_expand,
                kwargs={"target_line_1": line_1,
                        "root": root, "existing_ids": existing_ids,
                        "name_gen": name_gen},
                priority=importance * 1.3,
                affects_ddg=True,
            ))

    # ── Increment/decrement expansion (kept; already in relay form x++ → int t=1; x = x + t) ──
    for line_1 in {src_line, dst_line}:
        if line_1 is not None:
            candidates.append(TransformCandidate(
                name=f"unary_expand_L{line_1}",
                apply_fn=transform_unary_increment_expand,
                kwargs={"target_line_1": line_1,
                        "root": root, "existing_ids": existing_ids,
                        "name_gen": name_gen},
                priority=importance * 1.3,
                affects_ddg=True,
            ))

    # ── A.5 array-index extraction ──
    for line_1 in {src_line, dst_line}:
        if line_1 is not None:
            candidates.append(TransformCandidate(
                name=f"array_index_extract_L{line_1}",
                apply_fn=transform_array_index_extract,
                kwargs={"target_line_1": line_1,
                        "root": root, "existing_ids": existing_ids,
                        "name_gen": name_gen},
                priority=importance * 1.3,
                affects_ddg=True,
            ))

    # ── A.6 function-call argument extraction ──
    for line_1 in {src_line, dst_line}:
        if line_1 is not None:
            candidates.append(TransformCandidate(
                name=f"call_arg_extract_L{line_1}",
                apply_fn=transform_call_arg_extract,
                kwargs={"target_line_1": line_1,
                        "root": root, "existing_ids": existing_ids,
                        "name_gen": name_gen},
                priority=importance * 1.2,
                affects_ddg=True,
            ))

    # ── A.7 return-expression extraction (try both src/dst; return usually appears near the end) ──
    for line_1 in {src_line, dst_line}:
        if line_1 is not None:
            candidates.append(TransformCandidate(
                name=f"return_extract_L{line_1}",
                apply_fn=transform_return_extract,
                kwargs={"target_line_1": line_1,
                        "root": root, "existing_ids": existing_ids,
                        "name_gen": name_gen},
                priority=importance * 1.1,
                affects_ddg=True,
            ))

    # ── B.5 multi-variable declaration splitting ──
    candidates.append(TransformCandidate(
        name="multivar_decl_split",
        apply_fn=transform_multivar_decl_split,
        kwargs={"target_line_1": src_line,
                "root": root, "existing_ids": existing_ids},
        priority=importance * 0.8,
        affects_ddg=True,
    ))

    return candidates


def _build_cdg_transforms(
    lines: List[str], edge: dict, root: Node,
    existing_ids: Set[str],
) -> List[TransformCandidate]:
    """
    Generate all applicable transformation candidates for one CDG edge.

    After cleanup, only B.4 (ternary ↔ if-else conversion) is retained -- it is one of the few effective
    control-flow transformations that can still perturb both token sets and edge types. Other CDG transformations (for_to_while /
    while_to_dowhile / while_to_for / demorgan / split_and / split_or /
    dead_branch / early_return) show weak perturbation under sum-pooled token embedding,
    and have been deprecated.
    """
    candidates = []
    src_line = edge.get("src_line")
    dst_line = edge.get("dst_line")
    if src_line is None:
        return candidates

    importance = edge.get("importance", 0.5)

    # ── B.4 ternary ↔ if-else conversion (try both src/dst) ──
    for line_1 in {src_line, dst_line}:
        if line_1 is None:
            continue
        line_idx = line_1 - 1
        if line_idx < 0 or line_idx >= len(lines):
            continue
        # Register only when the source line contains "?" (rough filter; final validation is done inside the transform
        # with a strict tree-sitter check for conditional_expression)
        if "?" not in lines[line_idx]:
            continue
        candidates.append(TransformCandidate(
            name=f"ternary_to_if_L{line_1}",
            apply_fn=transform_ternary_to_if,
            kwargs={"target_line_1": line_1,
                    "root": root, "existing_ids": existing_ids},
            priority=importance * 1.2,
            affects_cdg=True,
            affects_ddg=True,
        ))

    return candidates


# ══════════════════════════════════════════════════════════════
#  Part 8 ─ Unified attack orchestration
# ══════════════════════════════════════════════════════════════

def attack_dependency_edges_ts(
    current_code_str: str,
    mapping,           # ExplanationMapping object
    wrapper,           # ModelWrapper instance
    true_label: int,
    tracker: RobustLineTracker,
    rename_map: dict,  # {old_name: new_name, ...}
    state,             # AttackState
    wv=None,           # gensim Word2Vec vocabulary (used by NameGenerator)
    max_attempts: int = 100,
    lang: str = "c",
    verbose: bool = True,
):
    """
    Unified dependency-edge attack entry point, combining data-flow + control-flow transformations.

    Strategy:
      1. Collect transformation candidates for all DDG / CDG edges
      2. Sort by importance × transform_weight
      3. Combine DDG + CDG transformations on the same line when possible
      4. Reparse the AST and update line-number mappings after each transformation
      5. Query the model after each transformation and check whether the prediction flips

    Args:
        current_code_str: current code (possibly already modified by the token stage)
        mapping:          ExplanationMapping, containing vulnerable_edges
        wrapper:          model wrapper
        true_label:       ground-truth label
        tracker:          line-number tracker
        rename_map:       variable-renaming map (dict form)
        state:            attack-state tracker
        wv:               gensim Word2Vec vocabulary (optional; used to generate embedding-friendly variable names)
        max_attempts:     maximum number of queries
        lang:             language identifier
        verbose:          whether to print details

    Returns:
        (success: bool, final_code: str)
    """
    from common.utils.gen_embedding import src2embedding
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attempts = 0
    lines = current_code_str.split('\n')

    # Collect all edges
    all_edges = []
    for edge in getattr(mapping, 'vulnerable_edges', []):
        edge_type = edge.get('edge_type', '')
        if edge_type in ('DDG', 'CDG'):
            all_edges.append(edge)

    if not all_edges:
        if verbose:
            print("  [TS] No available dependency edges")
        return False, current_code_str

    all_edges.sort(key=lambda e: e.get('importance', 0.5), reverse=True)

    # Initialize NameGenerator (requires the initial identifier set and W2V vocabulary)
    try:
        init_root = parse_code_to_ast(current_code_str, lang)
        init_ids = _collect_identifiers(init_root)
    except Exception:
        init_ids = set()

    name_gen = NameGenerator(wv, init_ids) if wv is not None else None

    # Inject MLM context into NameGenerator (consistent with attack_structure_guided)
    if name_gen is not None:
        try:
            from src.utils.gen_candidates import gen_candis, init_mlm, precompute_tokenize
            _mlm_singleton = init_mlm()

            def _code_provider():
                return '\n'.join(lines)

            _tok_cache = {"code": None, "precomputed": None}

            def _precomputed_provider():
                cur = _code_provider()
                if _tok_cache["code"] != cur:
                    try:
                        _tok_cache["precomputed"] = precompute_tokenize(cur)
                    except Exception:
                        _tok_cache["precomputed"] = None
                    _tok_cache["code"] = cur
                return _tok_cache["precomputed"]

            name_gen.mlm_ctx = {
                "gen_candis_fn": gen_candis,
                "mlm": _mlm_singleton,
                "code_provider": _code_provider,
                "precomputed_provider": _precomputed_provider,
            }
        except Exception as _e:
            if verbose:
                print(f"  [TS] Failed to inject MLM naming context ({_e}); falling back to W2V naming")

    applied_set: Set[Tuple[str, int, int, str]] = set()

    for round_idx in range(3):
        if attempts >= max_attempts:
            break

        for edge in all_edges:
            if attempts >= max_attempts:
                break

            edge_type = edge.get('edge_type', '')
            src_line_orig = edge.get('src_line')
            dst_line_orig = edge.get('dst_line')

            src_line_curr = tracker.resolve(src_line_orig) if src_line_orig else None
            dst_line_curr = tracker.resolve(dst_line_orig) if dst_line_orig else None

            code_str = '\n'.join(lines)
            try:
                root = parse_code_to_ast(code_str, lang)
            except Exception as e:
                if verbose:
                    print(f"  [TS] AST parsing failed: {e}")
                continue

            existing_ids = _collect_identifiers(root)
            if name_gen is not None:
                name_gen.existing_ids = existing_ids

            resolved_edge = dict(edge)
            if src_line_curr is not None:
                resolved_edge['src_line'] = src_line_curr
            if dst_line_curr is not None:
                resolved_edge['dst_line'] = dst_line_curr

            candidates = []
            if edge_type == 'DDG':
                candidates.extend(_build_ddg_transforms(
                    lines, resolved_edge, root, existing_ids, rename_map,
                    name_gen=name_gen,
                ))
            if edge_type == 'CDG':
                candidates.extend(_build_cdg_transforms(
                    lines, resolved_edge, root, existing_ids
                ))

            candidates.sort(key=lambda c: c.priority, reverse=True)

            for cand in candidates:
                if attempts >= max_attempts:
                    break

                edge_key = (edge_type, src_line_orig or 0, dst_line_orig or 0, cand.name)
                if edge_key in applied_set:
                    continue

                try:
                    new_lines, success, delta = cand.apply_fn(lines, **cand.kwargs)
                except Exception as e:
                    if verbose:
                        print(f"  [TS] Transform {cand.name} raised an exception: {e}")
                    continue

                if not success:
                    continue

                applied_set.add(edge_key)

                code_str = '\n'.join(new_lines)
                try:
                    proposed_data = src2embedding(
                        code_str.encode('utf-8'), true_label
                    ).to(DEVICE)
                    pred, true_conf = wrapper.predict_label_and_true_conf(
                        proposed_data, true_label
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [TS] Embedding/prediction failed: {e}")
                    continue

                attempts += 1
                state.update(code_str, pred, true_conf, true_label)

                if verbose:
                    conf_str = f"{true_conf:.4f}"
                    print(f"  [{attempts}] {cand.name} @ L{src_line_orig or '?'}"
                          f"→L{dst_line_orig or '?'} | conf={conf_str}"
                          f" | {'✓ FLIP!' if pred != true_label else '✗'}")

                if pred != true_label:
                    if delta != 0 and src_line_orig is not None:
                        target_curr = cand.kwargs.get('target_line_1',
                                      cand.kwargs.get('src_line_1', src_line_curr))
                        if target_curr is not None:
                            tracker.record_change(target_curr, 1, 1 + delta)
                    lines = new_lines
                    return True, '\n'.join(lines)

                if delta != 0 and src_line_orig is not None:
                    target_curr = cand.kwargs.get('target_line_1',
                                  cand.kwargs.get('src_line_1', src_line_curr))
                    if target_curr is not None:
                        tracker.record_change(target_curr, 1, 1 + delta)
                lines = new_lines
                break

    return False, '\n'.join(lines)


# ══════════════════════════════════════════════════════════════
#  Part 8b ─ Structure-transformation attack entry with global explainer ranking + safety checks
# ══════════════════════════════════════════════════════════════

def attack_structure_guided(
    current_code_str: str,
    mapping,           # ExplanationMapping
    wrapper,
    true_label: int,
    state,
    wv=None,
    max_attempts: int = 100,
    lang: str = "c",
    verbose: bool = True,
):
    """
    Unified structure-transformation attack (iterate by line importance).

    Strategy:
      Map all DDG/CDG edges back to lines, sort by maximum line importance,
      aggregate related edges line by line, and generate DDG/CDG transformation candidates.

    Safety: no global/local ERROR pre-check gate is used;
    rely on the safety guards inside each transformation function (_is_safe_for_insertion, etc.).
    Transformation functions naturally skip by returning success=False when the AST has issues.
    """
    from common.utils.gen_embedding import src2embedding
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attempts = 0
    lines = current_code_str.split('\n')
    tracker = RobustLineTracker(len(lines))

    try:
        root = parse_code_to_ast(current_code_str, lang)
    except Exception:
        root = None

    existing_ids = _collect_identifiers(root) if root is not None else set()
    name_gen = NameGenerator(wv, existing_ids) if wv is not None else None

    # ── Inject MLM context into NameGenerator (generate more "semantically reasonable" relay names based on current code) ──
    # Failure-tolerant: when MLM is unavailable, generate_one automatically falls back to the W2V path, preserving old behavior.
    if name_gen is not None:
        try:
            from src.utils.gen_candidates import gen_candis, init_mlm, precompute_tokenize
            _mlm_singleton = init_mlm()

            def _code_provider():
                return '\n'.join(lines)

            # MLM inference requires one tokenize preprocessing step; cache it on demand when the code string changes.
            _tok_cache = {"code": None, "precomputed": None}

            def _precomputed_provider():
                cur = _code_provider()
                if _tok_cache["code"] != cur:
                    try:
                        _tok_cache["precomputed"] = precompute_tokenize(cur)
                    except Exception:
                        _tok_cache["precomputed"] = None
                    _tok_cache["code"] = cur
                return _tok_cache["precomputed"]

            name_gen.mlm_ctx = {
                "gen_candis_fn": gen_candis,
                "mlm": _mlm_singleton,
                "code_provider": _code_provider,
                "precomputed_provider": _precomputed_provider,
            }
        except Exception as _e:
            if verbose:
                print(f"  [TS] Failed to inject MLM naming context ({_e}); falling back to W2V naming")

    applied_set: Set[str] = set()
    # Step A line-count limit: at most 25% of code lines
    max_step_a_lines = max(5, int(len(lines) * 0.25))
    step_a_count = 0


    # ═══════════════════════════════════════════════════
    # Step A: edge → line mapping, line-importance sorting, DDG/CDG transformations
    # ═══════════════════════════════════════════════════
    all_edges = (mapping.get_all_edges_ranked()
                 if hasattr(mapping, 'get_all_edges_ranked')
                 else sorted(getattr(mapping, 'all_edges', []),
                             key=lambda e: e.get('importance', 0), reverse=True))

    line_edge_map: Dict[int, dict] = {}
    for edge in all_edges:
        edge_type = edge.get('edge_type', '')
        if edge_type not in ('DDG', 'CDG'):
            continue
        imp = edge.get('importance', 0.0)
        for line_no in {edge.get('src_line'), edge.get('dst_line')}:
            if line_no is None:
                continue
            if line_no not in line_edge_map:
                line_edge_map[line_no] = {
                    'importance': imp,
                    'ddg_edges': [],
                    'cdg_edges': [],
                }
            else:
                line_edge_map[line_no]['importance'] = max(
                    line_edge_map[line_no]['importance'], imp
                )
            if edge_type == 'DDG':
                line_edge_map[line_no]['ddg_edges'].append(edge)
            else:
                line_edge_map[line_no]['cdg_edges'].append(edge)

    ranked_edge_lines = sorted(
        line_edge_map.items(),
        key=lambda x: x[1]['importance'],
        reverse=True,
    )

    if verbose:
        print(f"  [A] Edge-guided transforms: {len(all_edges)} edges -> {len(ranked_edge_lines)} target lines")

    if root is not None:
        for orig_line, line_info in ranked_edge_lines:
            if attempts >= max_attempts:
                break

            curr_line = tracker.resolve(orig_line)

            code_str = '\n'.join(lines)
            try:
                root = parse_code_to_ast(code_str, lang)
            except Exception:
                continue

            existing_ids = _collect_identifiers(root)
            if name_gen:
                name_gen.existing_ids = existing_ids

            candidates = []

            for edge in line_info['ddg_edges']:
                resolved_edge = dict(edge)
                src_orig = edge.get('src_line')
                dst_orig = edge.get('dst_line')
                if src_orig:
                    resolved_edge['src_line'] = tracker.resolve(src_orig)
                if dst_orig:
                    resolved_edge['dst_line'] = tracker.resolve(dst_orig)
                candidates.extend(_build_ddg_transforms(
                    lines, resolved_edge, root, existing_ids, {},
                    name_gen=name_gen,
                ))

            for edge in line_info['cdg_edges']:
                resolved_edge = dict(edge)
                src_orig = edge.get('src_line')
                dst_orig = edge.get('dst_line')
                if src_orig:
                    resolved_edge['src_line'] = tracker.resolve(src_orig)
                if dst_orig:
                    resolved_edge['dst_line'] = tracker.resolve(dst_orig)
                candidates.extend(_build_cdg_transforms(
                    lines, resolved_edge, root, existing_ids,
                ))

            # Deduplicate within the line and sort by priority
            seen_names = set()
            unique_candidates = []
            for cand in sorted(candidates, key=lambda c: c.priority, reverse=True):
                if cand.name not in seen_names:
                    seen_names.add(cand.name)
                    unique_candidates.append(cand)

            for cand in unique_candidates:
                if attempts >= max_attempts:
                    break

                cand_key = f"L{orig_line}_{cand.name}"
                if cand_key in applied_set:
                    continue

                try:
                    new_lines, success, delta = cand.apply_fn(lines, **cand.kwargs)
                except Exception:
                    continue

                if not success:
                    continue

                applied_set.add(cand_key)

                code_str = '\n'.join(new_lines)
                try:
                    proposed_data = src2embedding(
                        code_str.encode('utf-8'), true_label
                    ).to(DEVICE)
                    pred, true_conf = wrapper.predict_label_and_true_conf(
                        proposed_data, true_label
                    )
                except Exception:
                    continue

                attempts += 1
                state.update(code_str, pred, true_conf, true_label)

                if verbose:
                    print(f"  [A-{attempts}] {cand.name} @ L{orig_line}"
                          f" | conf={true_conf:.4f}"
                          f" | {'✓ FLIP!' if pred != true_label else '✗'}")

                if pred != true_label:
                    if delta != 0:
                        tc = cand.kwargs.get('target_line_1',
                             cand.kwargs.get('src_line_1', curr_line))
                        if tc:
                            tracker.record_change(tc, 1, 1 + delta)
                    lines = new_lines
                    return True, '\n'.join(lines)

                if delta != 0:
                    tc = cand.kwargs.get('target_line_1',
                         cand.kwargs.get('src_line_1', curr_line))
                    if tc:
                        tracker.record_change(tc, 1, 1 + delta)
                lines = new_lines
                step_a_count += 1
                break  # Accept at most one transformation per line

            if step_a_count >= max_step_a_lines:
                if verbose:
                    print(f"  [A] Line-count limit reached ({step_a_count}/{max_step_a_lines}); stopping")
                break  

    return False, '\n'.join(lines)



# ══════════════════════════════════════════════════════════════
#  Part 9 ─ Integration interface
# ══════════════════════════════════════════════════════════════

def create_tracker_from_code(code_str: str) -> RobustLineTracker:
    """Create a line-number tracker from a code string."""
    total = len(code_str.split('\n'))
    return RobustLineTracker(total)
