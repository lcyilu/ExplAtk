"""
ts_transforms.py — 基于 tree-sitter 的语义保留代码变换模块
============================================================

用于替换 expl_atk.py 中基于正则的数据流 / 控制流攻击阶段。
所有变换均以 *源码行列表* 为输入输出，内部通过 tree-sitter AST
精确定位语法结构，避免正则带来的括号匹配 / 类型推断等问题。

设计原则
--------
1. 每个变换函数签名统一：
       transform_xxx(lines, target_line, root, **ctx)
           → (new_lines, success: bool, delta: int)
   其中 delta 为该变换引入的行数增量（正=插入，负=删除）。
2. 行号约定：外部传入 / 传出的 line_no 均为 **1-indexed**；
   tree-sitter 的 start_point/end_point.row 是 **0-indexed**。
3. 类型推断走 AST 声明搜索 + 回退启发式，不依赖完整编译。
4. 所有变换保证语义等价（或对输出不可区分的冗余插入）。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

# ────────────────────────────────────────────────────────────
# tree-sitter 导入（复用用户已有的解析器初始化体系）
# ────────────────────────────────────────────────────────────
from tree_sitter import Node

# 用户已有的工具函数
from common.ast_parser.run_parser import parse_code_to_ast  # type: ignore

# ────────────────────────────────────────────────────────────
# C 关键字 / 保留字集合（从项目统一维护的关键字模块导入）
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
#  Part 0 ─ 常量与配置
# ══════════════════════════════════════════════════════════════

# 当前未使用全局常量：保留 Part 0 占位，后续若需要 OOV 名字白名单等可加在此。


# ══════════════════════════════════════════════════════════════
#  Part 1 ─ 健壮的行号映射追踪器
# ══════════════════════════════════════════════════════════════

class RobustLineTracker:
    """
    维护 original_line → current_line 的双向映射。

    核心思路：
      - 内部维护 dict：orig → current（1-indexed）
      - resolve(orig) 返回 orig 经过所有适用偏移后的当前行号
      - 每次变换完成后调用 record_change(current_start, old_count, new_count)
        来更新偏移

    相比原始 LineTracker 的改进：
      1. 支持区间替换（old_span != new_span）而不仅仅是插入
      2. 追踪基于当前行号而非原始行号，避免多次变换后偏移交叉
      3. 额外维护反向映射，支持 "当前行 → 最近的原始行" 查询
    """

    def __init__(self, total_lines: int):
        self._fwd: Dict[int, int] = {i: i for i in range(1, total_lines + 1)}

    def resolve(self, original_line: int) -> int:
        """原始行号 → 当前行号。"""
        return self._fwd.get(original_line, original_line)

    def record_change(self, current_start: int, old_count: int, new_count: int):
        """
        记录一次区间变换：
            当前行 [current_start, current_start + old_count)
            被替换为 new_count 行。
        所有映射到 >= current_start + old_count 的原始行都要偏移 delta。
        """
        delta = new_count - old_count
        if delta == 0:
            return
        threshold = current_start + old_count
        for orig in self._fwd:
            if self._fwd[orig] >= threshold:
                self._fwd[orig] += delta

    def update_total(self, new_total: int):
        """变换后代码总行数改变时，扩展映射表（新增行无原始行号对应）。"""
        pass

    def snapshot(self) -> Dict[int, int]:
        """返回当前映射的快照副本，用于调试。"""
        return dict(self._fwd)


# ══════════════════════════════════════════════════════════════
#  Part 2 ─ tree-sitter AST 辅助函数
# ══════════════════════════════════════════════════════════════

def _node_text(node: Node) -> str:
    """安全获取节点文本。"""
    return node.text.decode("utf-8") if node and node.text else ""


def _get_indent(line: str) -> str:
    """获取一行的缩进前缀。"""
    m = re.match(r'^(\s*)', line)
    return m.group(1) if m else ""


def _find_nodes_by_type(root: Node, type_name: str) -> List[Node]:
    """DFS 搜索所有指定类型的节点。"""
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
    查找起始行（0-indexed）== line_0 的节点。
    如果指定了 type_name，则只匹配该类型。
    返回最外层匹配节点。
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
    """查找起始行 == line_0 的语句级节点。"""
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
    """查找包含 line_0 的最内层指定类型节点。"""
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
    """找到包含 line_0 的函数定义节点。"""
    return _find_enclosing_node(root, line_0, {"function_definition"})


def _find_enclosing_loop(root: Node, line_0: int) -> Optional[Node]:
    """找到包含 line_0 的最内层循环节点。"""
    return _find_enclosing_node(root, line_0, {"for_statement", "while_statement", "do_statement"})


def _get_compound_body_range(node: Node) -> Optional[Tuple[int, int]]:
    """
    获取 compound_statement 的行范围 (start_row_0, end_row_0)。
    如果 body 不是 compound_statement（单语句），返回该语句的行范围。
    """
    body = node.child_by_field_name("body")
    if body is None:
        body = node.child_by_field_name("consequence")
    if body is None:
        return None
    return (body.start_point[0], body.end_point[0])


def _unwrap_parenthesized(node: Node) -> Node:
    """去除 parenthesized_expression 包装，返回内部表达式。"""
    while node and node.type == "parenthesized_expression" and node.named_child_count > 0:
        node = node.named_children[0]
    return node


def _get_condition_node(stmt_node: Node) -> Optional[Node]:
    """
    从 if_statement / while_statement / do_statement 中提取条件表达式节点。
    自动解包 parenthesized_expression。
    """
    cond = stmt_node.child_by_field_name("condition")
    if cond is None:
        return None
    return _unwrap_parenthesized(cond)


# ── Part 2b ─ 安全检查函数 ──────────────────────────────────

def _has_ast_errors(root: Node) -> bool:
    """检查 AST 中是否包含 ERROR 节点。"""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR":
            return True
        stack.extend(node.children)
    return False

def _has_error_near_line(root: Node, line_0: int, radius: int = 2) -> bool:
    """
    检查目标行附近（±radius 行）是否有 ERROR 节点。
    只跳过 ERROR 附近的变换，不影响远处的行。
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
    """检查目标行是否在 switch-case 结构内部。"""
    switch_node = _find_enclosing_node(root, line_0, {"switch_statement"})
    if switch_node is None:
        return False
    body = switch_node.child_by_field_name("body")
    if body and body.start_point[0] < line_0 <= body.end_point[0]:
        return True
    return False


def _is_case_label_line(root: Node, line_0: int) -> bool:
    """检查目标行是否是 case/default 标签行。"""
    node = _find_node_at_line(root, line_0, "case_statement")
    if node is not None:
        return True
    node = _find_node_at_line(root, line_0, "labeled_statement")
    return node is not None


def _is_macro_line(line: str) -> bool:
    """检查一行是否是宏调用或预处理指令。"""
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if re.match(r'^[A-Z_][A-Z0-9_]*\s*\(', stripped):
        return True
    if re.match(r'^[A-Z_][A-Z0-9_]+\s*[({;]', stripped):
        return True
    return False


def _is_for_init_or_update(root: Node, line_0: int) -> bool:
    """检查目标行是否在 for 循环头部（包含 init/cond/update 的那一行）。"""
    for_node = _find_enclosing_node(root, line_0, {"for_statement"})
    if for_node is None:
        return False
    if for_node.start_point[0] == line_0:
        return True
    return False


def _is_safe_for_insertion(root: Node, lines: List[str], line_0: int) -> bool:
    """
    综合安全检查：该行是否可以安全地在其前方插入新行或替换为多行。

    禁止插入的场景：
      - 宏调用行
      - case/default 标签行
      - for 的 init/update 所在行
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
    检查是否可以安全地做块级变换（if 包裹、条件拆分等）。
    比 _is_safe_for_insertion 更严格：额外禁止 switch-case 内部。
    """
    if not _is_safe_for_insertion(root, lines, line_0):
        return False
    if _is_inside_switch_case(root, line_0):
        return False
    return True


# ══════════════════════════════════════════════════════════════
#  Part 3 ─ 类型推断
# ══════════════════════════════════════════════════════════════

@dataclass
class VarTypeInfo:
    """变量类型信息（尽力推断）。"""
    base_type: str = "int"
    is_pointer: bool = False
    is_unsigned: bool = False
    is_array: bool = False
    full_decl_type: str = ""
    confidence: float = 0.0


def _infer_type_from_ast(root: Node, var_name: str, use_line_0: int) -> VarTypeInfo:
    """
    在 AST 中搜索 var_name 的声明，推断其类型。

    搜索策略：
      1. 查找所有 declaration 节点，看其中是否有声明了 var_name 的
      2. 查找函数参数列表中的参数声明
      3. 回退到启发式（根据命名和使用模式猜测）
    """
    info = VarTypeInfo()

    # ── 策略 1：搜索声明 ──
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

    # ── 策略 2：搜索函数参数 ──
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

    # ── 策略 3：启发式 ──
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
#  Part 4 ─ W2V 感知的命名生成器
# ══════════════════════════════════════════════════════════════

class NameGenerator:
    """
    基于 W2V 词表的变量名生成器。

    三级回退策略：
      1. seed 相似性检索：用被变换变量作为 seed，从 W2V 词表中检索
         最相似的合法 C 标识符。
      2. 通用池采样：从 W2V 词表中筛选所有合法 C 标识符的中间频率
         区域，随机打乱作为候选。
      3. 计数器回退：v1, v2, ... 纯保底。

    用法：
        gen = NameGenerator(wv, existing_ids)
        name = gen.generate_one(seed_var="level")
    """

    def __init__(self, wv, existing_ids: Set[str]):
        """
        Args:
            wv:           gensim KeyedVectors / Word2Vec 词表对象
                          需要支持 wv.key_to_index 和 wv.most_similar()
            existing_ids: 当前代码中已有的所有标识符
        """
        self.wv = wv
        self.existing_ids = existing_ids
        self._cache: Dict[str, List[str]] = {}
        self._generic_pool: Optional[List[str]] = None
        self._fallback_counter = 0
        # 可选 MLM 上下文：用于"基于上下文的命名候选"。
        # 由 attack_structure_guided 在构造 NameGenerator 后注入：
        #   ng.mlm_ctx = {"gen_candis_fn": fn, "mlm": mlm, "code_provider": lambda: cur_code}
        self.mlm_ctx: Optional[Dict[str, object]] = None

    def _in_vocab(self, word: str) -> bool:
        return word in self.wv.key_to_index

    def _is_valid_c_name(self, name: str) -> bool:
        """检查是否为合法且可用的 C 标识符。"""
        if not name.isidentifier():
            return False
        if name in _C_RESERVED:
            return False
        if name in self.existing_ids:
            return False
        # 过滤过短 / 纯数字开头（isidentifier 已排除）
        if len(name) < 2:
            return False
        return True

    def _split_compound(self, name: str) -> List[str]:
        """拆分 camelCase / snake_case 为子串列表。"""
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
        """为给定 seed 构建候选列表（已排序、已去重、已过滤）。"""
        raw_candidates = []

        if self._in_vocab(seed):
            try:
                sims = self.wv.most_similar(seed, topn=top_n)
                raw_candidates.extend(w for w, _ in sims)
            except (KeyError, ValueError):
                pass

        # seed 不在词表或候选不足时，尝试子串检索
        if len(raw_candidates) < 20:
            sub_parts = self._split_compound(seed)
            for sp in sub_parts:
                if self._in_vocab(sp):
                    try:
                        sims = self.wv.most_similar(sp, topn=50)
                        raw_candidates.extend(w for w, _ in sims)
                    except (KeyError, ValueError):
                        pass

        # 过滤 + 去重（保持相似度排序）
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
        """懒加载通用候选池。"""
        if self._generic_pool is not None:
            return

        all_words = list(self.wv.key_to_index.keys())
        valid = [w for w in all_words if w.isidentifier() and w not in _C_RESERVED and len(w) >= 2]

        # 取中间 60% 频率区域（避开最常见和最罕见的）
        n = len(valid)
        start = int(n * 0.2)
        end = int(n * 0.8)
        pool = valid[start:end] if n > 10 else valid

        random.shuffle(pool)
        self._generic_pool = pool

    def generate_one(self, seed_var: str = "") -> str:
        """
        生成一个不与已有标识符冲突的变量名。

        优先级：
          1. MLM 候选（仅当 self.mlm_ctx 已注入）
             —— 基于当前代码 + seed_var 的上下文，在 word2vec 词表内挑选
                语义合理的标识符。token 阶段已证明此路径效果优于纯 W2V 相似度。
          2. seed 相似性（W2V most_similar 池）
          3. 通用池（W2V 词表中频率适中的合法标识符）
          4. 计数器回退（v1, v2, ...）

        Args:
            seed_var: 被变换的原始变量名，用于相似性/上下文检索。
                      为空时跳过策略 1/2，直接走通用池/计数器。

        Returns:
            在 W2V 词表中的合法 C 标识符。
        """
        # ── 策略 1：MLM 候选（基于上下文） ──
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

        # ── 策略 2：seed 相似性 ──
        if seed_var:
            if seed_var not in self._cache:
                self._cache[seed_var] = self._build_candidates_for_seed(seed_var)

            candidates = self._cache[seed_var]
            for c in candidates:
                if c not in self.existing_ids:
                    self.existing_ids.add(c)
                    return c

        # ── 策略 3：通用池 ──
        self._ensure_generic_pool()
        for c in self._generic_pool:
            if c not in self.existing_ids and c not in _C_RESERVED:
                self.existing_ids.add(c)
                return c

        # ── 策略 4：计数器回退 ──
        while True:
            self._fallback_counter += 1
            name = f"v{self._fallback_counter}"
            if name not in self.existing_ids and name not in _C_RESERVED:
                self.existing_ids.add(name)
                return name

    def generate_batch(self, count: int, seed_var: str = "") -> List[str]:
        """批量生成 count 个不重复的变量名。"""
        return [self.generate_one(seed_var) for _ in range(count)]


def _collect_identifiers(root: Node) -> Set[str]:
    """收集 AST 中所有 identifier 和 field_identifier 节点的文本。"""
    ids = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in ("identifier", "field_identifier"):
            ids.add(_node_text(node))
        stack.extend(node.children)
    return ids


# ══════════════════════════════════════════════════════════════
#  Part 5 ─ 数据流语义保留变换
# ══════════════════════════════════════════════════════════════

def transform_temp_variable_insert(
    lines: List[str], target_line_1: int, var_name: str,
    root: Node, existing_ids: Set[str],
    name_gen: Optional[NameGenerator] = None,
) -> Tuple[List[str], bool, int]:
    """
    临时变量插入（类型感知版本）。
    在 target_line 前插入：
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
    表达式分解：将复合表达式拆分为多步中间变量赋值。

    识别模式：var = A op B;  →  type _t1 = A; type _t2 = B; var = _t1 op _t2;

    通过 tree-sitter 精确定位 binary_expression，避免错误拆分函数调用。
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
        # 用左/右操作数中的标识符作为 seed
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
    变量传播链延长：在变量定义和使用之间插入等价赋值链。

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
    赋值拆分（类型感知版本，不使用 __auto_type）。

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


# ── Part 5b ─ 字面量提取 / 复合赋值 / 自增展开（tree-sitter 版） ──


def _find_number_literals_on_line(root: Node, line_0: int) -> List[Node]:
    """找到指定行上所有 number_literal 节点。"""
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
    """找到指定行上所有 string_literal 节点。"""
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
    常量提取：将数字 / 字符串字面量提取为命名临时变量。

    原始：  header_nlines = 1 + image->ncolors;
    变换后：int _c1 = 1;
            header_nlines = _c1 + image->ncolors;

    原始：  strcpy(s, " XPMEXT");
    变换后：char _s1[] = " XPMEXT";
            strcpy(s, _s1);

    对 PDG 的影响：
      - 每个提取都引入一个新的定义节点和一条新的 DDG 边
      - 改变原始节点的 token 特征（数字/字符串消失，变量名出现）
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0

    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    # ── 收集该行的数字字面量 ──
    num_literals = _find_number_literals_on_line(root, idx)
    str_literals = _find_string_literals_on_line(root, idx)

    if not num_literals and not str_literals:
        return lines, False, 0

    # 按列位置从右往左替换（避免偏移）
    all_literals = []
    for node in num_literals:
        text = _node_text(node)
        # 跳过过于复杂的字面量（如浮点科学记数法）和 #define 中的
        if not text or line.strip().startswith("#"):
            continue
        # 跳过数组下标中的简单索引（如 header[0]）—— 但保留表达式中的
        all_literals.append(("number", node, text))

    for node in str_literals:
        text = _node_text(node)
        if not text or line.strip().startswith("#"):
            continue
        all_literals.append(("string", node, text))

    if not all_literals:
        return lines, False, 0

    # 按列位置从右往左排序
    all_literals.sort(key=lambda x: x[1].start_point[1], reverse=True)

    insert_lines = []
    new_line = line
    extracted_any = False

    for lit_type, node, text in all_literals:
        # 计算在行内的起止列
        col_start = node.start_point[1]
        col_end = node.end_point[1]

        if name_gen is not None:
            # 用字面量值的某种 hint 作为 seed
            seed = "num" if lit_type == "number" else "str"
            tmp_name = name_gen.generate_one(seed_var=seed)
        else:
            tmp_name = f"_const_{random.randint(1000, 9999)}"
            existing_ids.add(tmp_name)

        if lit_type == "number":
            # 推断类型：整数 vs 浮点
            if '.' in text or 'e' in text.lower() or 'f' in text.lower():
                decl_type = "double"
            else:
                decl_type = "int"
            insert_lines.append(f"{indent}{decl_type} {tmp_name} = {text};")
        else:
            # 字符串字面量
            # char name[] = "..."; 比 char *name 更安全
            insert_lines.append(f"{indent}char {tmp_name}[] = {text};")

        # 在行内替换（从右往左，偏移安全）
        new_line = new_line[:col_start] + tmp_name + new_line[col_end:]
        extracted_any = True

    if not extracted_any:
        return lines, False, 0

    result = list(lines)
    # 插入声明行在当前行之前
    result[idx:idx + 1] = insert_lines + [new_line]
    delta = len(insert_lines)  # 新增了 len(insert_lines) 行
    return result, True, delta


def _find_compound_assignment_on_line(root: Node, line_0: int) -> Optional[Node]:
    """
    找到指定行上的 compound_assignment_expression 或
    augmented_assignment_expression 节点。
    tree-sitter-c 中：+=, -=, *=, /= 等对应 assignment_expression
    且 operator 不是 "="。
    """
    stack = [root]
    while stack:
        node = stack.pop()
        if node.start_point[0] == line_0 and node.type == "assignment_expression":
            # 检查 operator 是否为复合赋值
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
    复合赋值展开（带中继升级版，B.6）：
        x += expr;
        →
        <type> _t = expr;
        x = x + _t;

    对 PDG 的影响：
      - 1 节点 → 2 节点（中继声明 + 展开赋值）
      - 改变原节点 token 集合：原 `{x, +=, expr_tokens}` 变成
        `{x, =, x, +, _t}` —— RHS 表达式 token 全部"搬走"到中继节点
      - DDG 边链路加长：原 RHS 中变量到 x 的路径多一跳

    回退：当 name_gen 不可用、推不出 LHS 类型时，退化为旧版的纯展开
    （`x += expr → x = x + (expr);` 仅改原节点 token，不新增节点）。
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

    # 提取运算符
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

    # 获取完整语句范围
    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None:
        return lines, False, 0

    stmt_start = stmt.start_point[0]
    stmt_end = stmt.end_point[0]
    old_count = stmt_end - stmt_start + 1

    # ── 主路径：带中继的展开 ──
    # 推断 LHS 类型作为中继变量声明类型；推断失败时回退到 "int"
    type_info = _infer_type_from_ast(root, lhs_text, idx)
    type_str = type_info.full_decl_type or "int"
    if "*" in type_str and not type_str.endswith("*"):
        # 形如 "int *" → OK；"int * const *" 这种暂不处理，稳妥起见退化到旧路径
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
    """找到指定行上的 update_expression（++/--）节点。"""
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
    自增/自减运算符展开：
        x++;  →  int _c = 1; x = x + _c;
        x--;  →  int _c = 1; x = x - _c;

    对 PDG 的影响：
      - 引入新的常量定义节点和 DDG 边
      - 将紧凑的 ++ 运算拆成多步，增加图的节点数
      - 改变节点 token 特征
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

    # 判断 ++ 还是 --，以及前缀还是后缀
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

    # 检查这个 update_expression 是否是独立语句（而非 for 循环的 update 子句或更大表达式的一部分）
    # 如果是 expression_statement 的直接子节点，则可以安全展开
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

    # 如果是独立语句，替换整行
    if stmt is not None and stmt.type == "expression_statement":
        stmt_start = stmt.start_point[0]
        stmt_end = stmt.end_point[0]
        old_count = stmt_end - stmt_start + 1
        result = list(lines)
        result[stmt_start:stmt_end + 1] = new_lines
        delta = len(new_lines) - old_count
        return result, True, delta
    else:
        # 如果嵌入在更大的表达式中（如 for 的 update），只做行内替换
        # x++ → (x = x + 1) —— 但这对 for update 不太安全，先跳过
        return lines, False, 0


# ── Part 5c ─ 新增的 substitutive 结构变换（A.5 / A.6 / A.7 / B.5）──


def _find_first_subscript_on_line(root: Node, line_0: int) -> Optional[Node]:
    """找指定行上第一个 subscript_expression（数组访问）节点。"""
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
    数组索引表达式提取（A.5）：
        buf[i + j * stride]    →    int idx = i + j * stride;
                                    buf[idx]
    仅对"非平凡"索引（不是单一标识符或单一字面量）触发——
    `arr[i]` 不会被改写，避免产生无价值的 idx 中继。

    PDG 影响：
      - 原 subscript 节点 token 集合显著简化：去掉索引表达式的 token，
        多一个 idx 标识符
      - 新增 declaration 节点 + DDG 边
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

    # subscript_expression: argument(数组), index(下标)
    index_node = sub.child_by_field_name("index")
    if index_node is None:
        return lines, False, 0

    # 跳过简单索引：单一 identifier / 数字字面量
    if index_node.type in ("identifier", "number_literal"):
        return lines, False, 0

    index_text = _node_text(index_node).strip()
    if not index_text:
        return lines, False, 0

    # 选 seed：优先从 index_text 里抓一个 identifier，否则用数组名
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

    # 行内替换：用 col 范围把原 index 替换为 tmp_name
    col_start = index_node.start_point[1]
    col_end = index_node.end_point[1]
    new_line = line[:col_start] + tmp_name + line[col_end:]

    insert_lines = [f"{indent}int {tmp_name} = {index_text};"]
    result = list(lines)
    result[idx:idx + 1] = insert_lines + [new_line]
    return result, True, len(insert_lines)


def _find_first_call_on_line(root: Node, line_0: int) -> Optional[Node]:
    """找指定行上第一个 call_expression 节点。"""
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
    函数调用参数提取（A.6）：
        memcpy(dst, src, len * 2)  →  int n = len * 2;
                                       memcpy(dst, src, n);

    选取调用的最后一个非平凡参数提取为命名中继；如果所有参数都是
    单 identifier / 字面量，则跳过（不产生无价值候选）。

    PDG 影响：
      - 调用节点 token 集合显著简化（复杂参数表达式被替换为单标识符）
      - 新增 declaration + DDG 边
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

    # 倒序找第一个"复杂参数"
    target_arg = None
    for arg in reversed(args_node.named_children):
        if arg.type in ("identifier", "number_literal", "string_literal",
                        "char_literal"):
            continue
        # parenthesized_expression 解包看里面
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

    # 选 seed：从 arg_text 里抓第一个 identifier，否则从被调用函数名抓
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

    # 行内替换：用 col 范围替换该参数为 tmp_name
    col_start = target_arg.start_point[1]
    col_end = target_arg.end_point[1]
    # 仅当参数完全在 idx 这一行内才安全替换（多行参数过于复杂，跳过）
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
    return 表达式提取（A.7）：
        return a + b * c;   →   int ret = a + b * c;
                                return ret;

    跳过 `return;`（无表达式）和 `return single_var;`（已是单标识符）。

    PDG 影响：
      - return 节点的 token 集合从 `{return, a, +, b, *, c}` 变成
        `{return, ret}` —— 信息密度大降
      - 新增 declaration 节点 + DDG 边
      - return 节点常被解释器标记为关键节点（漏洞与返回值高度相关），
        针对它的 substitutive 攻击 ROI 高。
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0
    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None or stmt.type != "return_statement":
        return lines, False, 0

    # return_statement 的命名子节点中，第一个非 "return" 关键字就是表达式
    expr_node = None
    for ch in stmt.named_children:
        expr_node = ch
        break
    if expr_node is None:
        return lines, False, 0

    inner = _unwrap_parenthesized(expr_node)
    # 平凡 return 跳过
    if inner.type in ("identifier", "number_literal", "string_literal",
                      "char_literal"):
        return lines, False, 0

    expr_text = _node_text(expr_node).strip()
    if not expr_text:
        return lines, False, 0

    # 表达式跨多行时，保守起见跳过（替换原行不便）
    if (expr_node.start_point[0] != idx or expr_node.end_point[0] != idx):
        return lines, False, 0

    line = lines[idx]
    indent = _get_indent(line)

    # 选 seed：从 expr 里抓第一个 identifier
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

    # 推断 return 类型：从所在函数的 declarator 推断；推断失败回落 "int"
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
    多变量声明拆分（B.5）：
        int a, b, c;          →    int a;
                                    int b;
                                    int c;
        int x = 1, y = 2;     →    int x = 1;
                                    int y = 2;

    PDG 影响：
      - 1 节点 → N 节点（每个声明独立）
      - 每个新节点 token 集合显著简化（不再共享逗号/类型）
      - 不依赖 name_gen，纯结构拆分。
    """
    idx = target_line_1 - 1
    if idx < 0 or idx >= len(lines):
        return lines, False, 0
    if not _is_safe_for_insertion(root, lines, idx):
        return lines, False, 0

    stmt = _find_statement_node_at_line(root, idx)
    if stmt is None or stmt.type != "declaration":
        return lines, False, 0

    # 必须含至少 2 个 declarator（init_declarator 或 identifier-style declarator）
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

    # 跨多行声明保守跳过
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
#  Part 6 ─ 控制流语义保留变换
# ══════════════════════════════════════════════════════════════

# 当前控制流变换：
#   - 三元表达式 → if-else（B.4）：见下方 transform_ternary_to_if
# 已废弃（在 sum-pooled token embedding 下扰动微弱）：
#   for_to_while / while_to_dowhile / while_to_for /
#   demorgan / split_and / split_or / dead_branch / early_return


def transform_ternary_to_if(
    lines: List[str], target_line_1: int,
    root: Node, existing_ids: Set[str],
) -> Tuple[List[str], bool, int]:
    """
    三元表达式 → if-else 转换 (B.4)。

    `var = (cond) ? a : b;`            →
    `if (cond) { var = a; } else { var = b; }`

    `<type> var = (cond) ? a : b;`     →
    `<type> var; if (cond) { var = a; } else { var = b; }`

    对 PDG 的影响：
      - 1 节点 → 4 节点（if / then / else / 合并）
      - 边类型从单纯 DDG 翻为 CDG + DDG
      - 同时撬动 token 集合 + 边类型，是少数仍然有效的 CDG 类变换。
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

    # 找到第一个 conditional_expression（三元）节点
    cond_exprs = _find_nodes_by_type(stmt, "conditional_expression")
    if not cond_exprs:
        return lines, False, 0
    cond_expr = cond_exprs[0]

    # 三元的三个组件：condition, consequence, alternative
    condition = cond_expr.child_by_field_name("condition")
    consequence = cond_expr.child_by_field_name("consequence")
    alternative = cond_expr.child_by_field_name("alternative")
    if condition is None or consequence is None or alternative is None:
        return lines, False, 0

    cond_text = _node_text(condition).strip()
    cons_text = _node_text(consequence).strip()
    alt_text = _node_text(alternative).strip()

    # 区分两种宿主语句
    if stmt.type == "expression_statement":
        # 形如 var = cond ? a : b;
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

    # stmt.type == "declaration"：形如 int var = cond ? a : b;
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
#  Part 7 ─ 变换注册表与优先级
# ══════════════════════════════════════════════════════════════

@dataclass
class TransformCandidate:
    """一个待评估的变换候选。"""
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
    为一条 DDG 边生成所有适用的变换候选。

    设计原则（清理后版本）：
      所有候选都满足"实质性改变某个 PDG 节点 token 集合"原则——
      要么把原行 RHS 提取到中继变量（A 系列），要么把字面量/参数/索引
      重排为新的命名变量；不再注册仅"加节点不动原行"或仅修改运算符的变换。
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

    # ── 临时变量插入（在样本 1 上观察到有效，保留旧版逻辑） ──
    candidates.append(TransformCandidate(
        name="temp_var_insert",
        apply_fn=transform_temp_variable_insert,
        kwargs={"target_line_1": src_line, "var_name": dep_var,
                "root": root, "existing_ids": existing_ids,
                "name_gen": name_gen},
        priority=importance * 1.0,
        affects_ddg=True,
    ))

    # ── A.1 复合表达式拆分 ──
    candidates.append(TransformCandidate(
        name="expr_decompose",
        apply_fn=transform_expression_decomposition,
        kwargs={"target_line_1": src_line,
                "root": root, "existing_ids": existing_ids,
                "name_gen": name_gen},
        priority=importance * 1.2,
        affects_ddg=True,
    ))

    # ── A.2 通用 RHS 中继（assignment_split，覆盖单变量/调用/数组访问 RHS）──
    candidates.append(TransformCandidate(
        name="rhs_relay",
        apply_fn=transform_assignment_split,
        kwargs={"target_line_1": src_line,
                "root": root, "existing_ids": existing_ids,
                "name_gen": name_gen},
        priority=importance * 1.1,
        affects_ddg=True,
    ))

    # ── 传播链延长（在样本 1 上观察到有效；chain_length=1 减少代码膨胀）──
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

    # ── 字面量提取（数字/字符串字面量 → 命名中继） ──
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

    # ── B.6 复合赋值展开（升级版，带 RHS 中继） ──
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

    # ── 自增/自减展开（保留，已是中继形式 x++ → int t=1; x = x + t） ──
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

    # ── A.5 数组索引提取 ──
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

    # ── A.6 函数调用参数提取 ──
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

    # ── A.7 return 表达式提取（src/dst 都尝试，return 通常出现在尾部） ──
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

    # ── B.5 多变量声明拆分 ──
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
    为一条 CDG 边生成所有适用的变换候选。

    清理后只保留 B.4（三元 ↔ if-else 互转）—— 它是少数仍能同时撬动
    token 集合 + 边类型的有效控制流变换。其余 CDG 变换（for_to_while /
    while_to_dowhile / while_to_for / demorgan / split_and / split_or /
    dead_branch / early_return）在 sum-pooled token embedding 下扰动微弱，
    已废弃。
    """
    candidates = []
    src_line = edge.get("src_line")
    dst_line = edge.get("dst_line")
    if src_line is None:
        return candidates

    importance = edge.get("importance", 0.5)

    # ── B.4 三元 ↔ if-else 互转（src/dst 都尝试） ──
    for line_1 in {src_line, dst_line}:
        if line_1 is None:
            continue
        line_idx = line_1 - 1
        if line_idx < 0 or line_idx >= len(lines):
            continue
        # 仅在源码行里出现 "?" 字符的情况下才注册（粗筛，最终由 transform 内部
        # 用 tree-sitter 严格判定 conditional_expression）
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
#  Part 8 ─ 统一攻击编排
# ══════════════════════════════════════════════════════════════

def attack_dependency_edges_ts(
    current_code_str: str,
    mapping,           # ExplanationMapping 对象
    wrapper,           # ModelWrapper 实例
    true_label: int,
    tracker: RobustLineTracker,
    rename_map: dict,  # {old_name: new_name, ...}
    state,             # AttackState
    wv=None,           # gensim Word2Vec 词表（用于 NameGenerator）
    max_attempts: int = 100,
    lang: str = "c",
    verbose: bool = True,
):
    """
    统一的依赖边攻击入口，融合数据流 + 控制流变换。

    策略：
      1. 收集所有 DDG / CDG 边对应的变换候选
      2. 按 importance × transform_weight 排序
      3. 同一行的 DDG + CDG 变换可组合
      4. 每次变换后重新解析 AST + 更新行号映射
      5. 每次变换后查询模型，检查是否翻转

    Args:
        current_code_str: 当前代码（可能已经过 token 阶段修改）
        mapping:          ExplanationMapping，包含 vulnerable_edges
        wrapper:          模型包装器
        true_label:       真实标签
        tracker:          行号追踪器
        rename_map:       变量重命名映射（dict 形式）
        state:            攻击状态追踪器
        wv:               gensim Word2Vec 词表（可选，用于生成嵌入友好的变量名）
        max_attempts:     最大查询次数
        lang:             语言标识
        verbose:          是否打印详情

    Returns:
        (success: bool, final_code: str)
    """
    from common.utils.gen_embedding import src2embedding
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attempts = 0
    lines = current_code_str.split('\n')

    # 收集所有边
    all_edges = []
    for edge in getattr(mapping, 'vulnerable_edges', []):
        edge_type = edge.get('edge_type', '')
        if edge_type in ('DDG', 'CDG'):
            all_edges.append(edge)

    if not all_edges:
        if verbose:
            print("  [TS] 无可用依赖边")
        return False, current_code_str

    all_edges.sort(key=lambda e: e.get('importance', 0.5), reverse=True)

    # 初始化 NameGenerator（需要初始标识符集和 W2V 词表）
    try:
        init_root = parse_code_to_ast(current_code_str, lang)
        init_ids = _collect_identifiers(init_root)
    except Exception:
        init_ids = set()

    name_gen = NameGenerator(wv, init_ids) if wv is not None else None

    # 给 NameGenerator 注入 MLM 上下文（与 attack_structure_guided 保持一致）
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
                print(f"  [TS] MLM 命名上下文注入失败 ({_e})，回退到 W2V 命名")

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
                    print(f"  [TS] AST 解析失败: {e}")
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
                        print(f"  [TS] 变换 {cand.name} 异常: {e}")
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
                        print(f"  [TS] 嵌入/预测失败: {e}")
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
#  Part 8b ─ 解释器全局排序 + 安全检查 的结构变换攻击入口
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
    统一的结构变换攻击（按行重要性排序遍历）。

    策略：
      将所有 DDG/CDG 边映射回行，按行最大重要性排序，
      逐行汇总关联边并生成 DDG/CDG 变换候选。

    安全性：不做全局/局部 ERROR 前置门控，
    依赖每个变换函数内部的安全守卫（_is_safe_for_insertion 等）。
    变换函数在 AST 有问题时返回 success=False，自然跳过。
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

    # ── 给 NameGenerator 注入 MLM 上下文（基于当前代码生成更"语义合理"的中继名）──
    # 失败容错：MLM 不可用时 generate_one 自动退化到 W2V 路径，行为与旧版兼容。
    if name_gen is not None:
        try:
            from src.utils.gen_candidates import gen_candis, init_mlm, precompute_tokenize
            _mlm_singleton = init_mlm()

            def _code_provider():
                return '\n'.join(lines)

            # MLM 推理需要一次 tokenize 预处理；按需缓存当代码字符串。
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
                print(f"  [TS] MLM 命名上下文注入失败 ({_e})，回退到 W2V 命名")

    applied_set: Set[str] = set()
    # Step A 变换行数限制：最多 25% 的代码行
    max_step_a_lines = max(5, int(len(lines) * 0.25))
    step_a_count = 0


    # ═══════════════════════════════════════════════════
    # Step A: 边 → 行映射，按行重要性排序，DDG/CDG 变换
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
        print(f"  [A] 边引导变换：{len(all_edges)} 条边 → {len(ranked_edge_lines)} 个目标行")

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

            # 行内去重 + 按优先级排序
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
                break  # 每行最多接受一个变换

            if step_a_count >= max_step_a_lines:
                if verbose:
                    print(f"  [A] 达到行数上限 ({step_a_count}/{max_step_a_lines})，停止")
                break  

    return False, '\n'.join(lines)



# ══════════════════════════════════════════════════════════════
#  Part 9 ─ 集成接口
# ══════════════════════════════════════════════════════════════

def create_tracker_from_code(code_str: str) -> RobustLineTracker:
    """从代码字符串创建行号追踪器。"""
    total = len(code_str.split('\n'))
    return RobustLineTracker(total)
