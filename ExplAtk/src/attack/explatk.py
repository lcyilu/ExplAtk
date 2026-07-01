"""
ExplAtk: 基于解释器引导的迭代式对抗攻击
==========================================

攻击流程：
  阶段一: GA Token 级攻击（遗传算法 + 解释器 importance 引导）
  阶段二: DDG/CDG 边引导的结构变换攻击
"""

import re
import random
import torch
import sys
from pathlib import Path

_METHOD_ROOT = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_METHOD_ROOT, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common.attack_result import AttackResult, to_text
from common.utils.gen_embedding import (
    renamed_pdg_to_embedding,
    src2pdg,
    src2embedding,
    load_word_vectors,
    pdg2embedding,
    multi_renamed_pdg_to_embedding,
)
from common.utils.renamer import rename_identifier,rename_identifiers
from src.model.wrapper import ModelWrapper
from src.utils.gen_candidates import gen_candis_w2v,init_mlm,gen_candis, gen_candis_codet5, precompute_tokenize, precompute_tokenize_codet5
from src.utils.parser import extract_identifiers_from_one_src
from src.attack.ts_transforms import (
    attack_dependency_edges_ts,
    attack_structure_guided,
    create_tracker_from_code,
    RobustLineTracker,
)
from src.pre.attack_trace import AttackTraceLogger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MLM = init_mlm()

# ──────────────────────────────────────────────────────────────────────
# 消融实验默认模式（改这里就能切换全局默认行为，无需改上游调用）
#   "token_only"  : Explain(v0) → Token
#   "struct_only" : Explain(v0) → 结构变换
#   "full"        : Explain(v0) → Token → Re-Explain → 结构变换  （论文主线）
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODE = "token_only"

# ──────────────────────────────────────────────────────────────────────
# Adaptive Re-Explanation 阈值
#   conf_drop = original_true_conf - best_true_conf  （token 阶段把 true_conf 压低了多少）
#   - drop < DELTA_LOW   : token 几乎没动模型 → 解释信号也不会有大变化，跳过 re-explain
#   - drop > DELTA_HIGH  : token 已经把 true_conf 压得很低 → 接近翻转，省下解析开销直接做结构变换
#   - 其余区间           : 做 re-explain（信息增量最大的"甜区"）
# 关闭自适应时（adaptive_reexplain=False），不论 drop 多少都做 re-explain。
# ──────────────────────────────────────────────────────────────────────
DELTA_LOW = 0.05
DELTA_HIGH = 0.40

# ──────────────────────────────────────────────────────────────────────
# 解释器选择开关
# ──────────────────────────────────────────────────────────────────────
# 修改这一处即可全局切换解释器（不改任何调用方）。
# 可选值: "coca" | "robust"
DEFAULT_EXPLAINER = "robust"

# ──────────────────────────────────────────────────────────────────────
# Token GA guidance 选择开关
# ──────────────────────────────────────────────────────────────────────
# 默认 explanation，保证上游批量实验不改调用、不增加 masking 查询开销。
# 可选值: "explanation" | "random" | "masking"
DEFAULT_GUIDANCE_MODE = "explanation"
VALID_GUIDANCE_MODES = {"explanation", "random", "masking"}

from explain.coca_explainer import CocaExplainer
from explain.robust_explainer import RobustExplainer
from explain.mapping import map_explanation_to_source,ExplanationMapping
from src.model.wrapper import ModelWrapper
from src.utils.gen_embedding import read_json


# ══════════════════════════════════════════════════════════════════════
# 基础设施
# ══════════════════════════════════════════════════════════════════════

class LineTracker:
    """跨阶段行号偏移追踪。"""

    def __init__(self):
        self.offsets = []

    def record(self, original_line, delta):
        self.offsets.append((original_line, delta))

    def resolve(self, original_line):
        actual = original_line
        for ref_line, delta in self.offsets:
            if original_line > ref_line:
                actual += delta
        return actual


class RenameMap:
    """跨阶段变量重命名追踪，供数据流阶段解析 dep_variable。"""

    def __init__(self):
        self.mapping = {}

    def add(self, old_name, new_name):
        self.mapping[old_name] = new_name

    def resolve(self, var_name):
        current = var_name
        visited = set()
        while current in self.mapping and current not in visited:
            visited.add(current)
            current = self.mapping[current]
        return current


class AttackState:
    """
    全程追踪攻击过程中的关键变体：
      best_variant:          true_conf 最低的变体（置信度下降最大）
      first_success_variant: 第一个翻转预测的变体
      final_variant:         最终变体
    """

    def __init__(self, original_true_conf):
        self.original_true_conf = original_true_conf
        self.best_variant = None
        self.best_true_conf = original_true_conf
        self.first_success_variant = None
        self.success_true_conf = None
        self.final_variant = None
        self.final_pred = None
        self.final_true_conf = None

    def update(self, code_str, pred, true_conf, true_label):
        if true_conf < self.best_true_conf:
            self.best_true_conf = true_conf
            self.best_variant = code_str
        if self.first_success_variant is None and pred != true_label:
            self.first_success_variant = code_str
            self.success_true_conf = true_conf
        self.final_variant = code_str
        self.final_pred = pred
        self.final_true_conf = true_conf


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════

def _negate_simple_expr(expr):
    """对简单表达式取反。"""
    expr = expr.strip()
    if expr.startswith('!'):
        return expr[1:].strip()
    ops = [('==', '!='), ('!=', '=='), ('<=', '>'), ('>=', '<'),
           ('<', '>='), ('>', '<=')]
    for old_op, new_op in ops:
        if old_op in expr:
            return expr.replace(old_op, new_op, 1)
    return f'!({expr})'


def _to_str(code):
    """统一转为 str。"""
    if isinstance(code, bytes):
        return code.decode('utf-8')
    return code


def _query_from_lines(lines, wrapper, true_label, state):
    """从代码行列表构建嵌入并查询模型，更新状态。返回 (是否翻转, 代码字符串)。"""
    code_str = '\n'.join(lines)
    proposed_data = src2embedding(code_str.encode('utf-8'), true_label).to(DEVICE)
    pred, true_conf = wrapper.predict_label_and_true_conf(proposed_data, true_label)
    state.update(code_str, pred, true_conf, true_label)
    return pred != true_label, code_str

def _get_explainer(expl_name, wrapper:ModelWrapper):
    explainer_switch = {
        'coca': CocaExplainer(model=wrapper.model, device=DEVICE),
        'robust': RobustExplainer(model=wrapper.model, device=DEVICE),
    }
    return explainer_switch.get(expl_name.lower(), None)


def _should_reexplain(original_true_conf, best_true_conf,
                      delta_low=DELTA_LOW, delta_high=DELTA_HIGH):
    """
    Adaptive Re-Explanation 触发判据。

    Args:
        original_true_conf: 原始代码上的 true-label 置信度
        best_true_conf:     token 阶段所达到的最低 true-label 置信度
        delta_low / delta_high: 上下阈值（来自模块级常量，可被参数覆盖）

    Returns:
        (do_reexplain: bool, reason: str)
    """
    drop = original_true_conf - best_true_conf
    if drop < delta_low:
        return False, (
            f"drop={drop:.4f} < δ_low={delta_low:.2f}，token 阶段几乎未撼动模型，"
            f"重新解释信息增量低，跳过"
        )
    if drop > delta_high:
        return False, (
            f"drop={drop:.4f} > δ_high={delta_high:.2f}，token 阶段已显著压低置信度，"
            f"省下解析开销直接做结构变换"
        )
    return True, f"drop={drop:.4f} ∈ [{delta_low:.2f}, {delta_high:.2f}]，处于信息增量甜区"


def _identifier_lines(source_code, identifiers):
    """返回每个 identifier 在源码中出现过的行号，供不同 guidance 结果格式保持一致。"""
    source_text = source_code if isinstance(source_code, str) else source_code.decode('utf-8')
    lines = source_text.split('\n')
    line_map = {}
    for ident in identifiers:
        pattern = re.compile(r'\b' + re.escape(ident) + r'\b')
        line_map[ident] = [idx + 1 for idx, line in enumerate(lines) if pattern.search(line)]
    return line_map


def _random_identifier_importance(source_code, lang='c'):
    """Random guidance：只随机化 identifier 优先级，不额外查询模型。"""
    code_bytes = source_code.encode('utf-8') if isinstance(source_code, str) else source_code
    raw_ids = extract_identifiers_from_one_src(code_bytes, lang=lang)
    unique_ids = list(set(raw_ids))
    if not unique_ids:
        return []

    line_map = _identifier_lines(source_code, unique_ids)
    result = [
        {
            'name': ident,
            'importance': random.random(),
            'lines': line_map.get(ident, []),
        }
        for ident in unique_ids
    ]
    result.sort(key=lambda x: x['importance'], reverse=True)
    return result


def _masking_identifier_importance(current_code_str, current_pdg, wrapper: ModelWrapper,
                                   wv, true_label, lang='c', verbose=False):
    """
    Masking guidance：参考 masking.py，用把 identifier 替换为 MASK 后的 true-label
    confidence drop 作为 importance。仅在 guidance_mode="masking" 时调用。
    """
    from src.config import MASK

    code_bytes = current_code_str.encode('utf-8') if isinstance(current_code_str, str) else current_code_str
    raw_ids = extract_identifiers_from_one_src(code_bytes, lang=lang)
    unique_ids = list(set(raw_ids))
    if not unique_ids:
        return []

    ori_data = pdg2embedding(current_pdg, wv, true_label).to(DEVICE)
    ori_prob = wrapper.predict_prob(ori_data, true_label)
    line_map = _identifier_lines(current_code_str, unique_ids)

    result = []
    if verbose:
        print(f"[GA] Masking guidance: scoring {len(unique_ids)} identifiers")
    for ident in unique_ids:
        proposed_data = renamed_pdg_to_embedding(
            current_pdg, wv, ident, MASK, true_label
        ).to(DEVICE)
        score = wrapper.compute_importance(true_label, ori_prob, proposed_data)
        result.append({
            'name': ident,
            'importance': float(score),
            'lines': line_map.get(ident, []),
        })

    result.sort(key=lambda x: x['importance'], reverse=True)
    return result


def _get_guided_identifier_importance(guidance_mode, current_code_str, current_pdg,
                                      mapping, wrapper: ModelWrapper, wv,
                                      true_label, lang='c', verbose=False):
    """按 guidance_mode 生成统一的 [{'name', 'importance', 'lines'}] 排序列表。"""
    mode = (guidance_mode or DEFAULT_GUIDANCE_MODE).lower()
    if mode not in VALID_GUIDANCE_MODES:
        raise ValueError(
            f"guidance_mode 必须是 {sorted(VALID_GUIDANCE_MODES)}，收到 {guidance_mode!r}"
        )

    if mode == "explanation":
        return mapping.get_identifier_importance(current_code_str, lang)
    if mode == "random":
        return _random_identifier_importance(current_code_str, lang)
    return _masking_identifier_importance(
        current_code_str, current_pdg, wrapper, wv, true_label, lang, verbose=verbose
    )


# ══════════════════════════════════════════════════════════════════════
# 阶段一：Token 级攻击
# ══════════════════════════════════════════════════════════════════════

def _attack_token(current_code_str, mapping, wrapper: ModelWrapper, wv, true_label, 
                  rename_map, state, lang, max_attempts):
    """
    Token 级攻击：使用 Word2Vec 寻找候选词，通过修改 PDG 节点属性进行快速评估。
    """
    current_pdg = src2pdg(current_code_str)
    attempts = 0

    # 1. 初始化环境：提取代码中所有可替换的标识符
    raw_ids = extract_identifiers_from_one_src(current_code_str.encode('utf-8'), lang)
    unique_identifiers = list(set(raw_ids))
    existing_identifiers = set(raw_ids)

    # 2. 遍历解释器识别出的关键（Vulnerable）节点
    for node in mapping.vulnerable_nodes:
        if attempts >= max_attempts:
            break

        line_no = node['line_no']
        lines = current_code_str.split('\n')
        line_idx = line_no - 1
        
        if line_idx < 0 or line_idx >= len(lines):
            continue

        line_content = lines[line_idx]

        # 筛选当前行中包含的可替换标识符
        line_targets = [
            v for v in unique_identifiers
            if re.search(r'\b' + re.escape(v) + r'\b', line_content)
        ]
        print(f"Targeting identifiers in line {line_no}: {line_targets}")

        # 3. 对当前行的每个变量尝试替换
        for target_var in line_targets:
            if attempts >= max_attempts:
                break

            # 生成 W2V 候选词并过滤掉已存在的标识符
            candidates = gen_candis_w2v(target_var, wv, top_k=5)
            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in existing_identifiers]
            if not valid_candidates:
                continue

            # 4. 尝试每个候选词
            for id_candi in valid_candidates:
                if attempts >= max_attempts:
                    break

                # 使用快速路径：直接修改 PDG 节点特征，无需重新生成 CPG/PDG
                proposed_data = renamed_pdg_to_embedding(
                    current_pdg, wv, target_var, id_candi, true_label
                ).to(DEVICE)

                pred, true_conf = wrapper.predict_label_and_true_conf(
                    proposed_data, true_label
                )
                attempts += 1

                # 生成替换后的源码字符串
                proposed_code = _to_str(
                    rename_identifier(current_code_str, target_var, id_candi, lang)
                )
                
                # 更新全局攻击状态记录
                state.update(proposed_code, pred, true_conf, true_label)

                # 如果标签成功翻转，直接返回成功结果
                if pred != true_label:
                    rename_map.add(target_var, id_candi)
                    return True, proposed_code

            # 5. 累积扰动处理：
            # 如果本轮所有候选词都未能翻转标签，默认保留第一个候选词（贪心累积）
            # 注意：此处可根据建议改为“仅保留能降低置信度的最佳候选”
            best = valid_candidates[0]
            current_code_str = _to_str(
                rename_identifier(current_code_str, target_var, best, lang)
            )
            
            # 应用累积扰动后，必须重新解析 PDG，以确保下一轮替换在正确的图结构上进行
            current_pdg = src2pdg(current_code_str)
            rename_map.add(target_var, best)

            # 更新当前代码的标识符集合
            existing_identifiers.discard(target_var)
            existing_identifiers.add(best)
            unique_identifiers = [best if v == target_var else v for v in unique_identifiers]

    return False, current_code_str

def _attack_token_drop(current_code_str, mapping, wrapper: ModelWrapper, wv, true_label,
                       rename_map, state, lang, max_attempts):
    """
    Token 级攻击：
    优先选择能显著降低 margin 的替换；
    若没有降低 margin 的候选，则默认保留第一个合法候选。

    margin = true_conf - other_conf

    margin 越小，越接近攻击成功；
    margin < 0 时，说明 other_label 分数已经超过 true_label，预测发生翻转。
    """
    current_pdg = src2pdg(current_code_str)
    attempts = 0

    # 获取初始 margin 作为基准
    initial_data = pdg2embedding(current_pdg, wv, true_label).to(torch.device(DEVICE))

    _, current_baseline_conf, current_baseline_margin = \
        wrapper.predict_label_and_true_conf_margin(initial_data, true_label)

    raw_ids = extract_identifiers_from_one_src(
        current_code_str.encode('utf-8'), lang
    )
    unique_identifiers = list(set(raw_ids))
    existing_identifiers = set(raw_ids)

    # 缓存 tokenize 结果，代码变化时失效重算
    _cached_code_for_tokenize = None
    _cached_precomputed = None

    for node in mapping.vulnerable_nodes:
        if attempts >= max_attempts:
            break

        line_no = node['line_no']
        lines = current_code_str.split('\n')
        line_idx = line_no - 1

        if line_idx < 0 or line_idx >= len(lines):
            continue

        line_content = lines[line_idx]

        line_targets = [
            v for v in unique_identifiers
            if re.search(r'\b' + re.escape(v) + r'\b', line_content)
        ]

        for target_var in line_targets:
            if attempts >= max_attempts:
                break

            # 复用 tokenize 结果：只有代码变化时重新 tokenize
            if _cached_code_for_tokenize != current_code_str:
                _cached_code_for_tokenize = current_code_str
                _cached_precomputed = precompute_tokenize(current_code_str)

            # candidates = gen_candis_w2v(target_var, wv, top_k=5)
            candidates = gen_candis(current_code_str, MLM, target_var, _precomputed=_cached_precomputed)
            # candidates = gen_candis_codet5(current_code_str, MLM, target_var, _precomputed=_cached_precomputed)
            if not candidates:
                continue

            valid_candidates = [
                c for c in candidates
                if c not in existing_identifiers
            ]

            if not valid_candidates:
                continue

            best_candi_for_var = None
            min_margin_for_var = current_baseline_margin
            best_true_conf_for_var = current_baseline_conf

            # 记录已评估候选的结果，方便 fallback 时更新 baseline
            evaluated_results = {}

            for id_candi in valid_candidates:
                if attempts >= max_attempts:
                    break

                # 快速评估：使用 renamed_pdg_to_embedding 避免频繁重新解析 PDG
                proposed_data = renamed_pdg_to_embedding(
                    current_pdg, wv, target_var, id_candi, true_label
                ).to(torch.device(DEVICE))

                pred, true_conf, margin = \
                    wrapper.predict_label_and_true_conf_margin(
                        proposed_data, true_label
                    )

                attempts += 1

                proposed_code_tmp = _to_str(
                    rename_identifier(
                        current_code_str, target_var, id_candi, lang
                    )
                )

                # 存储结构保持不变：仍然记录 true_conf
                state.update(proposed_code_tmp, pred, true_conf, true_label)

                evaluated_results[id_candi] = {
                    "pred": pred,
                    "true_conf": true_conf,
                    "margin": margin,
                }

                # 如果成功翻转标签，直接返回
                if pred != true_label:
                    rename_map.add(target_var, id_candi)
                    return True, proposed_code_tmp

                # 如果没翻转，寻找能使 margin 下降最多的候选
                if margin < min_margin_for_var:
                    min_margin_for_var = margin
                    best_true_conf_for_var = true_conf
                    best_candi_for_var = id_candi

            # 如果已经用完 query budget，并且没有评估到任何候选，就不再强行应用 fallback
            if attempts >= max_attempts and not evaluated_results:
                break

            # 优先保留降低 margin 最多的候选；
            # 如果没有任何候选降低 margin，则默认保留第一个合法候选。
            if best_candi_for_var is not None:
                chosen_candi = best_candi_for_var
                current_baseline_margin = min_margin_for_var
                current_baseline_conf = best_true_conf_for_var
            else:
                chosen_candi = valid_candidates[0]

                print(
                    f"No candidate reduced margin for {target_var}, "
                    f"fallback to first candidate: {chosen_candi}"
                )

                # fallback 候选如果已经评估过，直接用它的结果更新 baseline
                if chosen_candi in evaluated_results:
                    current_baseline_margin = evaluated_results[chosen_candi]["margin"]
                    current_baseline_conf = evaluated_results[chosen_candi]["true_conf"]
                else:
                    # 极少数情况：chosen_candi 没被评估，比如中途达到 max_attempts
                    # 这里不额外消耗 query，只保留旧 baseline
                    pass

            current_code_str = _to_str(
                rename_identifier(
                    current_code_str, target_var, chosen_candi, lang
                )
            )

            # 应用累积扰动后，重新生成 PDG
            current_pdg = src2pdg(current_code_str)

            rename_map.add(target_var, chosen_candi)

            existing_identifiers.discard(target_var)
            existing_identifiers.add(chosen_candi)

            unique_identifiers = [
                chosen_candi if v == target_var else v
                for v in unique_identifiers
            ]

    return False, current_code_str

def _attack_token_genetic(
    current_code_str: str,
    mapping,              # ExplanationMapping
    wrapper: ModelWrapper,
    wv,
    true_label: int,
    rename_map,           # RenameMap
    state,                # AttackState
    lang: str = 'c',
    max_queries: int = 100,
    pop_size: int = 20,
    max_generations: int = 15,
    top_k_candidates: int = 5,
    elite_count: int = 2,
    crossover_rate: float = 0.7,
    base_mutation_rate: float = 0.15,
    batch_eval_size: int = 0,
    verbose: bool = True,
    trace_logger=None,
    guidance_mode: str = DEFAULT_GUIDANCE_MODE,
):
    """
    基于遗传算法的 Token 级攻击。

    染色体编码：
      - 每个基因对应一个可替换标识符
      - 基因值 0 = 不替换，1..K = 使用第 k 个 W2V 候选

    适应度：
      - margin = true_conf - other_conf，越小越好
      - margin < 0 表示预测翻转（攻击成功）

    Guidance 融合：
      - guidance_mode 决定 identifier importance 来源：explanation / random / masking
      - 标识符的 importance 分数影响初始化和变异概率
      - 高 importance 标识符更可能被替换、更频繁变异
      - 低 importance 标识符作为兜底，仍有机会参与进化

    Args:
        current_code_str:   当前源码字符串
        mapping:            ExplanationMapping 对象
        wrapper:            ModelWrapper 实例
        wv:                 gensim Word2Vec 词表
        true_label:         真实标签
        rename_map:         RenameMap 对象（记录跨阶段重命名）
        state:              AttackState 对象
        lang:               语言标识
        max_queries:        最大查询预算
        pop_size:           种群大小
        max_generations:    最大进化代数
        top_k_candidates:   每个标识符的 W2V 候选数
        elite_count:        精英保留数
        crossover_rate:     交叉概率
        base_mutation_rate: 基础变异概率
        batch_eval_size:    批量评估大小（0=逐个评估）
        verbose:            是否打印详情
        guidance_mode:      identifier guidance 来源，默认 explanation

    Returns:
        (success: bool, final_code: str)
    """
    import random as rnd

    current_pdg = src2pdg(current_code_str)

    # ═══════════════════════════════════════════════════════════
    # Step 1: 构建标识符表 + 候选词表 + 重要性权重
    # ═══════════════════════════════════════════════════════════

    guidance_mode = (guidance_mode or DEFAULT_GUIDANCE_MODE).lower()

    # 按 guidance_mode 获取每个标识符的重要性。
    # explanation 为默认路径；random 不查询模型；masking 才会额外执行 masking importance 查询。
    ranked_identifiers = _get_guided_identifier_importance(
        guidance_mode, current_code_str, current_pdg,
        mapping, wrapper, wv, true_label, lang, verbose=verbose,
    )

    if not ranked_identifiers:
        if verbose:
            print("[GA] 无可替换标识符")
        return False, current_code_str

    # 提取当前代码中已有的所有标识符（用于冲突检测）
    existing_ids = set(item['name'] for item in ranked_identifiers)

    # 为每个标识符生成 W2V 候选并过滤
    identifiers = []     # 标识符名称列表
    candidates = []      # 对应的候选词列表（不含原名）
    importances = []     # 对应的重要性分数

    # 预计算：对同一份代码只 tokenize 一次，N 个变量共用结果
    _precomputed = precompute_tokenize(current_code_str)

    for item in ranked_identifiers:
        name = item['name']
        imp = item['importance']

        # candis = gen_candis_w2v(name, wv, top_k=top_k_candidates)
        candis = gen_candis(current_code_str, MLM, name, _precomputed=_precomputed)
        # candis = gen_candis_codet5(current_code_str, MLM, name, _precomputed=_precomputed)
        if not candis:
            continue

        # 过滤掉已存在的标识符
        valid = [c for c in candis if c not in existing_ids]
        if not valid:
            continue

        identifiers.append(name)
        candidates.append(valid)
        importances.append(imp)

    num_genes = len(identifiers)
    if num_genes == 0:
        if verbose:
            print("[GA] 所有标识符均无合法候选")
        return False, current_code_str

    # 归一化重要性到 [0, 1]
    max_imp = max(importances) if max(importances) > 0 else 1.0
    norm_importances = [imp / max_imp for imp in importances]

    if verbose:
        print(f"[GA] guidance={guidance_mode}, 标识符数={num_genes}, "
              f"种群={pop_size}, 最大代数={max_generations}")
        top3 = [(identifiers[i], f"{importances[i]:.4f}") for i in range(min(3, num_genes))]
        print(f"[GA] Top-3 标识符: {top3}")

    queries_used = 0

    def trace_replacements(chromosome):
        """构造当前染色体中实际替换的 identifier 级 trace 信息。"""
        records = []
        for i, gene in enumerate(chromosome):
            if gene > 0 and gene <= len(candidates[i]):
                records.append({
                    'identifier': identifiers[i],
                    'candidate': candidates[i][gene - 1],
                    'candidate_index': gene - 1,
                    'importance': float(importances[i]),
                    'normalized_importance': float(norm_importances[i]),
                    'rank': i + 1,
                })
        return records

    # ═══════════════════════════════════════════════════════════
    # Step 2: 染色体操作辅助函数
    # ═══════════════════════════════════════════════════════════

    def decode(chromosome):
        """将染色体解码为 {old_name: new_name} 的替换字典。"""
        rename_dict = {}
        for i, gene in enumerate(chromosome):
            if gene > 0 and gene <= len(candidates[i]):
                rename_dict[identifiers[i]] = candidates[i][gene - 1]
        return rename_dict

    # 添加缓存，避免相同子代重新判断
    fitness_cache = {}

    def evaluate(chromosome, generation=None, individual_index=None):
        """
        评估一条染色体的适应度。

        Returns:
            (margin, pred, true_conf, proposed_code_str)
            margin 越小越好，< 0 表示翻转
        """
        key = tuple(chromosome)
        if key in fitness_cache:
            return fitness_cache[key]
        nonlocal queries_used

        rename_dict = decode(chromosome)
        if not rename_dict:
            return 999.0, true_label, 1.0, current_code_str

        # 用 multi_renamed_pdg_to_embedding 快速评估
        proposed_data = multi_renamed_pdg_to_embedding(
            current_pdg, wv, rename_dict, true_label
        ).to(DEVICE)

        pred, true_conf, margin = wrapper.predict_label_and_true_conf_margin(
            proposed_data, true_label
        )
        queries_used += 1

        # 生成对应的源码（用于记录和返回）
        proposed_code = _to_str(
            rename_identifiers(current_code_str,rename_dict)
        )

        # 更新全局攻击状态
        state.update(proposed_code, pred, true_conf, true_label)

        # 记录每次实际模型查询对应的替换和解释分数，供后续脆弱空间分析。
        # trace 默认关闭时不构造 replacements，避免正常批量攻击产生额外开销。
        if getattr(trace_logger, 'enabled', False):
            trace_logger.log_query(
                phase='token_ga',
                generation=generation,
                individual_index=individual_index,
                local_query_index=queries_used,
                global_query_count=wrapper.get_query_count(),
                replacements=trace_replacements(chromosome),
                pred=pred,
                true_label=true_label,
                true_conf=true_conf,
                margin=margin,
                success=(pred != true_label),
                guidance_mode=guidance_mode,
            )

        result = (margin, pred, true_conf, proposed_code)
        fitness_cache[key] = result
        return result

    def init_chromosome():
        """
        生成一条初始染色体。
        高 importance 标识符有更大概率被初始化为替换状态。
        """
        chromosome = [0] * num_genes
        for i in range(num_genes):
            # 初始化替换概率 = 0.3 + 0.5 * normalized_importance
            # importance=1.0 → 80% 概率被替换
            # importance=0.0 → 30% 概率被替换
            p_init = 0.3 + 0.5 * norm_importances[i]
            if rnd.random() < p_init:
                chromosome[i] = rnd.randint(1, len(candidates[i]))
        return chromosome

    def mutate(chromosome):
        """
        变异操作：importance 加权的变异概率。
        """
        result = list(chromosome)
        for i in range(num_genes):
            # 变异概率 = base_rate * (1 + importance)
            p_mut = base_mutation_rate * (1.0 + norm_importances[i])
            if rnd.random() < p_mut:
                result[i] = rnd.randint(0, len(candidates[i]))
        return result

    def crossover(parent_a, parent_b):
        """均匀交叉：每个基因位独立地从两个父代中选择。"""
        child = [0] * num_genes
        for i in range(num_genes):
            if rnd.random() < 0.5:
                child[i] = parent_a[i]
            else:
                child[i] = parent_b[i]
        return child

    def tournament_select(population, fitnesses, k=3):
        """锦标赛选择：随机抽 k 个，选最优。"""
        indices = rnd.sample(range(len(population)), min(k, len(population)))
        best_idx = min(indices, key=lambda x: fitnesses[x])
        return population[best_idx]

    # ═══════════════════════════════════════════════════════════
    # Step 3: GA 主循环
    # ═══════════════════════════════════════════════════════════

    # 初始化种群
    population = [init_chromosome() for _ in range(pop_size)]

    best_ever_margin = 999.0
    best_ever_chromosome = None
    best_ever_code = current_code_str

    for gen in range(max_generations):
        if queries_used >= max_queries:
            break

        # ── 评估当代种群 ──
        fitnesses = []
        for idx, chrom in enumerate(population):
            if queries_used >= max_queries:
                fitnesses.append(999.0)
                continue

            margin, pred, true_conf, proposed_code = evaluate(
                chrom, generation=gen, individual_index=idx
            )
            fitnesses.append(margin)

            # 更新全局最优
            if margin < best_ever_margin:
                best_ever_margin = margin
                best_ever_chromosome = list(chrom)
                best_ever_code = proposed_code

            # 攻击成功：立即返回
            if pred != true_label:
                if verbose:
                    rename_dict = decode(chrom)
                    renames = ', '.join(f'{k}→{v}' for k, v in rename_dict.items())
                    print(f"[GA] ✓ 代{gen} 个体{idx} 翻转成功！"
                          f" margin={margin:.4f} 查询={queries_used}")
                    print(f"[GA]   替换: {renames}")

                # 记录重命名到 rename_map
                for old_n, new_n in decode(chrom).items():
                    rename_map.add(old_n, new_n)

                return True, proposed_code

        if queries_used >= max_queries:
            break

        # ── 打印当代统计 ──
        if verbose:
            gen_best = min(fitnesses)
            gen_avg = sum(f for f in fitnesses if f < 999) / max(1, sum(1 for f in fitnesses if f < 999))
            num_replacing = sum(1 for g in population[fitnesses.index(gen_best)] if g > 0)
            print(f"[GA] 代{gen}: best_margin={gen_best:.4f} "
                  f"avg={gen_avg:.4f} 替换数={num_replacing} "
                  f"查询={queries_used}/{max_queries}")

        # ── 构建下一代 ──
        new_population = []

        # 精英保留
        sorted_indices = sorted(range(len(fitnesses)), key=lambda x: fitnesses[x])
        for i in range(min(elite_count, len(sorted_indices))):
            new_population.append(list(population[sorted_indices[i]]))

        # 交叉 + 变异生成剩余个体
        while len(new_population) < pop_size:
            if rnd.random() < crossover_rate:
                p1 = tournament_select(population, fitnesses)
                p2 = tournament_select(population, fitnesses)
                child = crossover(p1, p2)
            else:
                child = list(tournament_select(population, fitnesses))

            child = mutate(child)
            new_population.append(child)

        population = new_population

        # ── 提前终止：连续 3 代最优 margin 未改善 ──
        if gen >= 3:
            # 简单的停滞检测
            pass  # 可选：记录历史最优，连续无改善则 break

    # ═══════════════════════════════════════════════════════════
    # Step 4: GA 结束，返回最优个体对应的代码
    # ═══════════════════════════════════════════════════════════

    if best_ever_chromosome is not None:
        rename_dict = decode(best_ever_chromosome)
        for old_n, new_n in rename_dict.items():
            rename_map.add(old_n, new_n)

        if verbose:
            num_replaced = sum(1 for g in best_ever_chromosome if g > 0)
            print(f"[GA] ✗ 未翻转, best_margin={best_ever_margin:.4f} "
                  f"替换数={num_replaced} 总查询={queries_used}")

    return False, best_ever_code



# ══════════════════════════════════════════════════════════════════════
# 结果构建
# ══════════════════════════════════════════════════════════════════════

def _build_attack_result(sample_id, true_label, original_code, state,
                         original_pred, original_true_conf, wrapper):
    """从 AttackState 收集信息构建 AttackResult。"""
    final_variant = state.final_variant or original_code
    best_variant = state.best_variant or original_code
    final_pred = state.final_pred if state.final_pred is not None else original_pred
    final_true_conf = state.final_true_conf if state.final_true_conf is not None else original_true_conf

    is_attackable = original_pred == true_label
    success = is_attackable and final_pred != true_label

    return AttackResult(
        sample_id=sample_id,
        attack_name="expl_atk",
        model_name=wrapper.model_name,
        true_label=true_label,
        original_pred=original_pred,
        original_true_conf=original_true_conf,
        is_attackable=is_attackable,
        success=success,
        query_count=wrapper.get_query_count(),
        original_code=to_text(original_code),
        final_variant=to_text(final_variant),
        best_variant_by_conf_drop=to_text(best_variant),
        first_success_variant=to_text(state.first_success_variant) if state.first_success_variant else None,
        final_pred=final_pred,
        final_true_conf=final_true_conf,
        best_true_conf=state.best_true_conf,
        success_true_conf=state.success_true_conf,
    )


# ══════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════

def expl_atk(
    sample_id,
    original_code,
    true_label,
    mapping,
    wrapper: ModelWrapper,
    wv,
    lang='c',
    max_token_attempts=250,
    max_dependency_attempts=100,
    verbose=True,
    # ── 消融开关（默认 "full"=token→reexplain→struct，对应论文主线）──
    mode: str = DEFAULT_MODE,
    reexplain_fn=None,
    # Adaptive Re-Explanation 开关
    adaptive_reexplain: bool = True,
    # query-level attack trace 输出目录；None 时使用 attack_trace.py 的默认目录
    trace_dir=None,
):
    """
    ExplAtk: 基于解释器引导的迭代式对抗攻击。

    Args:
        sample_id:                样本 ID
        original_code:            原始源码（str 或 bytes）
        true_label:               真实标签
        mapping:                  ExplanationMapping 对象（针对 original_code）
        wrapper:                  ModelWrapper 实例
        wv:                       gensim Word2Vec 词表
        lang:                     语言标识（默认 'c'）
        max_token_attempts:       Token 阶段最大尝试次数
        max_dependency_attempts:  结构变换阶段最大尝试次数
        verbose:                  是否打印过程

        mode: 消融模式，三选一：
          - "token_only"  : Explain(v0) → Token        （只跑 token 阶段）
          - "struct_only" : Explain(v0) → 结构变换       （只跑结构变换）
          - "full"        : Explain(v0) → Token → Re-Explain(v1) → 结构变换
        reexplain_fn: callable(code_str) -> ExplanationMapping
                      仅在 mode="full" 时被调用一次，将基于 token 阶段产出的
                      变体重新生成 mapping。失败/未提供时沿用旧 mapping。
        adaptive_reexplain: 当 mode="full" 时是否启用 Adaptive Re-Explanation。
                            True：依据 conf_drop 自适应决定是否真正调用 reexplain_fn
                                  （详见 _should_reexplain 与模块常量 DELTA_LOW/HIGH）；
                            False：无条件 re-explain（用于消融对比）。

    Returns:
        AttackResult
    """
    assert mode in ("token_only", "struct_only", "full"), \
        f"mode 必须是 'token_only' / 'struct_only' / 'full'，收到 {mode!r}"

    wrapper.reset_query_count()
    ori_code_str = original_code if isinstance(original_code, str) else original_code.decode('utf-8')

    # ── 原始预测 ──
    from common.utils.gen_embedding import src2embedding
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_data = src2embedding(ori_code_str.encode('utf-8'), true_label).to(DEVICE)
    original_pred, original_true_conf = wrapper.predict_label_and_true_conf(
        original_data, true_label
    )

    if original_pred != true_label:
        if verbose:
            print(f"[ExplAtk] 样本 {sample_id}: 预测 {original_pred} ≠ 真实 {true_label}，跳过")
        state = AttackState(original_true_conf)
        return _build_attack_result(
            sample_id, true_label, ori_code_str, state,
            original_pred, original_true_conf, wrapper,
        )

    state = AttackState(original_true_conf)
    rename_map = RenameMap()
    current_code = ori_code_str
    current_mapping = mapping
    guidance_mode = DEFAULT_GUIDANCE_MODE
    trace_logger = AttackTraceLogger(
        sample_id=sample_id,
        model_name=wrapper.model_name,
        attack_name="expl_atk",
        trace_dir=trace_dir,
        original_true_conf=original_true_conf,
        guidance_mode=guidance_mode,
    )

    run_token = mode in ("token_only", "full")
    run_struct = mode in ("struct_only", "full")
    run_reexplain = (mode == "full")

    if verbose:
        print(f"[ExplAtk] 模式: {mode} (token={run_token}, "
              f"reexplain={run_reexplain}, struct={run_struct})")

    # ════════════════════════════════════════════════════════════
    # 阶段一：GA Token 级攻击
    # ════════════════════════════════════════════════════════════
    if run_token:
        if verbose:
            print(f"[ExplAtk] 阶段一：GA Token 攻击 (预算={max_token_attempts})")

        success, current_code = _attack_token_genetic(
            current_code, current_mapping, wrapper, wv, true_label,
            rename_map, state, lang,
            max_queries=max_token_attempts,
            pop_size=10,
            max_generations=20,
            verbose=verbose,
            trace_logger=trace_logger,
            guidance_mode=guidance_mode,
        )

        if success:
            if verbose:
                renames = ', '.join(f'{k}→{v}' for k, v in rename_map.mapping.items())
                print(f"  ✓ Token 攻击成功！{renames}  查询: {wrapper.get_query_count()}")
            trace_logger.close({
                'stage': 'token',
                'success': True,
                'query_count': wrapper.get_query_count(),
                'best_true_conf': state.best_true_conf,
            })
            return _build_attack_result(
                sample_id, true_label, ori_code_str, state,
                original_pred, original_true_conf, wrapper,
            )

        if verbose:
            print(f"  ✗ Token 未成功（查询: {wrapper.get_query_count()}）")

        # ── 回退到 token 阶段中置信度最低的变体 ──
        if (state.best_variant is not None
                and state.best_true_conf < state.final_true_conf):
            current_code = state.best_variant
            if verbose:
                print(f"  ↩ 回退到最优变体 (conf={state.best_true_conf:.4f}"
                      f" < final={state.final_true_conf:.4f})")

    # token_only 模式：到此结束
    if not run_struct:
        trace_logger.close({
            'stage': 'token_only',
            'success': state.final_pred is not None and state.final_pred != true_label,
            'query_count': wrapper.get_query_count(),
            'best_true_conf': state.best_true_conf,
        })
        return _build_attack_result(
            sample_id, true_label, ori_code_str, state,
            original_pred, original_true_conf, wrapper,
        )

    # ════════════════════════════════════════════════════════════
    # 中间步骤：在 token 与 结构变换 之间做一次 re-explain
    # 仅 mode="full" 触发；struct_only 直接跳过（沿用 v0 的 mapping）。
    # 当 adaptive_reexplain=True 时，按 conf_drop 自适应决定是否实际执行。
    # ════════════════════════════════════════════════════════════
    if run_reexplain:
        # —— Adaptive 早停判定 —————————————————————————————
        skip_by_adaptive = False
        if adaptive_reexplain:
            do_re, reason = _should_reexplain(
                state.original_true_conf, state.best_true_conf,
            )
            if verbose:
                tag = "执行" if do_re else "跳过"
                print(f"[ExplAtk] Adaptive Re-Explain → {tag}: {reason}")
            skip_by_adaptive = not do_re

        if skip_by_adaptive:
            pass  # 不调用 reexplain_fn，沿用旧 mapping
        elif reexplain_fn is None:
            if verbose:
                print("[ExplAtk] Re-Explain 跳过：未提供 reexplain_fn，沿用旧 mapping")
        else:
            if verbose:
                print("[ExplAtk] Re-Explain：基于 token 阶段产出的变体重新解释")
            try:
                new_mapping = reexplain_fn(current_code)
                if new_mapping is not None:
                    current_mapping = new_mapping
                    if verbose:
                        print(f"  ✓ 已刷新 mapping "
                              f"(关键节点数={len(current_mapping.vulnerable_nodes)})")
                else:
                    if verbose:
                        print("  ✗ reexplain_fn 返回 None，沿用旧 mapping")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Re-Explain 失败 ({e})，沿用旧 mapping")

    # ════════════════════════════════════════════════════════════
    # 阶段二：DDG/CDG 边引导的结构变换攻击
    # ════════════════════════════════════════════════════════════
    if verbose:
        print(f"[ExplAtk] 阶段二：结构变换攻击")

    success, current_code = attack_structure_guided(
        current_code_str=current_code,
        mapping=current_mapping,
        wrapper=wrapper,
        true_label=true_label,
        state=state,
        wv=wv,
        max_attempts=max_dependency_attempts,
        lang=lang,
        verbose=verbose,
    )


    if verbose:
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {status}  总查询: {wrapper.get_query_count()}")

    trace_logger.close({
        'stage': 'final',
        'success': bool(success),
        'query_count': wrapper.get_query_count(),
        'best_true_conf': state.best_true_conf,
    })
    return _build_attack_result(
        sample_id, true_label, ori_code_str, state,
        original_pred, original_true_conf, wrapper,
    )

def demo_atk(source_path):
    with open(source_path, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()
    result = run_expl_attack(
        expl_name=DEFAULT_EXPLAINER,
        source_path=source_path,
        source_code=source_code,
        model_name="reveal",
        checkpoint_path="{HOME_PATH}/vul_explain/23_explain_eval_ISSTA/trained_model/ori-ds/reveal/reveal-cwe119/mod_94.59_92.5_96.77_93.61.ckpt",
        true_label=1,
    )

    if result.success:
        print(f"\n{'─' * 60}")
        print("首次成功变体:")
        print(f"{'─' * 60}")
        print(result.first_success_variant)


def replace_path_part(path, old_part, new_part):
    """
    只替换路径中的某一级目录名，避免误替换文件名中的字符串。
    """
    path = Path(path)
    parts = list(path.parts)

    try:
        idx = parts.index(old_part)
    except ValueError:
        raise ValueError(f"路径中没有目录 {old_part}: {path}")

    parts[idx] = new_part
    return Path(*parts)


def source_to_related_paths(source_path):
    """
    根据 source_path 自动判断 normal / ori，并转换得到：
        dot_path
        cpg_bin_path
        json_path

    规则：
    1. normal:
        normal-src -> normal-pdg
        normal-src -> normal-cpg-bin
        normal-src -> normal-embedding

    2. ori:
        BigVul/all-src -> BigVul/ori-pdg
        BigVul/all-src -> BigVul/ori-cpg-bin
        BigVul/all-src -> BigVul/ori-embedding

        其他目录/src -> ori-pdg
        其他目录/src -> ori-cpg-bin
        其他目录/src -> ori-embedding

    3. 后缀：
        .c -> .dot
        .c -> .bin
        .c -> .json
    """

    source_path = Path(source_path)

    if source_path.suffix != ".c":
        raise ValueError(f"当前只支持 .c 文件，但收到: {source_path}")

    parts = source_path.parts

    # case 1: normal
    if "normal-src" in parts:
        dot_path = replace_path_part(
            source_path, "normal-src", "normal-pdg"
        ).with_suffix(".dot")

        cpg_bin_path = replace_path_part(
            source_path, "normal-src", "normal-cpg-bin"
        ).with_suffix(".bin")

        json_path = replace_path_part(
            source_path, "normal-src", "normal-embedding"
        ).with_suffix(".json")

    # case 2: ori in BigVul
    elif "BigVul" in parts and "all-src" in parts:
        dot_path = replace_path_part(
            source_path, "all-src", "ori-pdg"
        ).with_suffix(".dot")

        cpg_bin_path = replace_path_part(
            source_path, "all-src", "ori-cpg-bin"
        ).with_suffix(".bin")

        json_path = replace_path_part(
            source_path, "all-src", "ori-embedding"
        ).with_suffix(".json")

    # case 3: ori in other datasets
    elif "src" in parts:
        dot_path = replace_path_part(
            source_path, "src", "ori-pdg"
        ).with_suffix(".dot")

        cpg_bin_path = replace_path_part(
            source_path, "src", "ori-cpg-bin"
        ).with_suffix(".bin")

        json_path = replace_path_part(
            source_path, "src", "ori-embedding"
        ).with_suffix(".json")

    else:
        raise ValueError(
            f"无法根据路径自动判断类型，路径中应包含 normal-src、BigVul/all-src 或 src: {source_path}"
        )

    return str(dot_path), str(cpg_bin_path), str(json_path)


def run_expl_attack(
    expl_name,
    source_path,
    model_name,
    checkpoint_path,
    source_code,
    true_label,
    sample_id="unknown",
    sample_i=None,
    lang='cpp',
    input_dim=100,
    output_dim=200,
    # ── 消融开关：透传给 expl_atk ──
    mode: str = DEFAULT_MODE,
    trace_dir=None,
):
    effective_sample_id = sample_id if sample_id != "unknown" else (
        sample_i if sample_i is not None else "unknown"
    )
    wrapper = ModelWrapper(model_name, checkpoint_path, input_dim=input_dim, output_dim=output_dim)
    wv = load_word_vectors() 
    explainer = _get_explainer(expl_name,wrapper)

    dot_path, cpg_bin_path, json_path =  source_to_related_paths(source_path)
    data = read_json(json_path)
    pred_label = wrapper.predict_label(data)  # 通常为 1
    result = explainer.explain(data, pred_label)

    # 2. 映射回源码（完整信息）
    mapping = map_explanation_to_source(
        explain_result=result,
        dot_path=dot_path,
        cpg_bin_path=cpg_bin_path,
        source_path=source_path,     # 可选
    )

    mapping.print_summary()

    # ── 构造 re-explain 闭包：仅 mode="full" 时被 expl_atk 调用一次 ──
    # 闭包捕获 explainer / pred_label，对新代码就近重生成 dot+cpg-bin。
    def _reexplain_fn(code_str: str):
        """
        对当前变体重新生成 PDG/CPG-bin，再跑一次 explainer，
        返回新的 ExplanationMapping。任何中间环节失败都抛异常，
        由 expl_atk 内部 try/except 兜底（沿用旧 mapping）。
        """
        import os, tempfile
        from common.utils.gen_embedding import (
            joern_parse, joern_export, src2embedding,
        )

        with tempfile.TemporaryDirectory(prefix="reexplain_") as tmp_dir:
            tmp_src = os.path.join(tmp_dir, "tmp.c")
            tmp_bin = os.path.join(tmp_dir, "tmp.bin")
            tmp_pdg_dir = os.path.join(tmp_dir, "tmp_pdg")
            tmp_dot = os.path.join(tmp_dir, "tmp_pdg.dot")

            with open(tmp_src, "w", encoding="utf-8") as f:
                f.write(code_str if isinstance(code_str, str) else code_str.decode("utf-8"))

            # 生成 cpg-bin 与 dot
            joern_parse(tmp_src, tmp_bin)
            if not os.path.exists(tmp_bin):
                raise RuntimeError("joern_parse 未产出 cpg-bin")
            joern_export(tmp_bin, tmp_pdg_dir)
            if not os.path.exists(tmp_dot):
                raise RuntimeError("joern_export 未产出 dot")

            # 跑 explainer：基于变体代码的 embedding
            new_data = src2embedding(
                code_str.encode("utf-8") if isinstance(code_str, str) else code_str,
                pred_label,
            ).to(DEVICE)
            new_result = explainer.explain(new_data, pred_label)

            new_mapping = map_explanation_to_source(
                explain_result=new_result,
                dot_path=tmp_dot,
                cpg_bin_path=tmp_bin,
                source_path=None,  # 用变体源码代替不再准确的原 source_path
            )
        return new_mapping

    return expl_atk(
        sample_id=sample_id,
        original_code=source_code,
        true_label=pred_label,
        mapping=mapping,
        wrapper=wrapper,
        wv=wv,
        lang='c',
        max_token_attempts=250,
        max_dependency_attempts=100,
        verbose=True,
        mode=mode,
        reexplain_fn=_reexplain_fn,
        trace_dir=trace_dir,
    )

if __name__ == '__main__':
    demo_atk("{HOME_PATH}/VulDS/BigVul/all-src/vul/1_CVE-2013-1788_poppler_CWE-119_bbc2d8918fe234b7ef2c480eb148943922cc0959_1.c")