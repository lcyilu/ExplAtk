"""
ExplAtk: Explainer-guided iterative adversarial attack
=======================================================

Attack workflow:
  Stage 1: GA token-level attack (genetic algorithm + explainer importance guidance)
  Stage 2: Structure transformation attack guided by DDG/CDG edges
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
from src.utils.attack_trace import AttackTraceLogger

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MLM = init_mlm()

# ──────────────────────────────────────────────────────────────────────
# Default ablation mode (change here to switch global default behavior without changing upstream calls)
#   "token_only"  : Explain(v0) → Token
#   "struct_only" : Explain(v0) → structure transformation
#   "full"        : Explain(v0) → Token → Re-Explain → structure transformation (main paper pipeline)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MODE = "token_only"

# ──────────────────────────────────────────────────────────────────────
# Adaptive Re-Explanation thresholds
#   conf_drop = original_true_conf - best_true_conf  (how much the token stage lowers true_conf)
#   - drop < DELTA_LOW   : the token stage barely changes the model → explanation signals are unlikely to change much, so skip re-explain
#   - drop > DELTA_HIGH  : the token stage already lowers true_conf substantially → close to flipping, so skip parsing overhead and run structure transformation directly
#   - otherwise          : run re-explain (the information-gain sweet spot)
# When adaptation is disabled (adaptive_reexplain=False), always run re-explain regardless of drop.
# ──────────────────────────────────────────────────────────────────────
DELTA_LOW = 0.05
DELTA_HIGH = 0.40

# ──────────────────────────────────────────────────────────────────────
# Explainer selection switch
# ──────────────────────────────────────────────────────────────────────
# Change this setting to switch the explainer globally without changing any callers.
# Options: "coca" | "robust"
DEFAULT_EXPLAINER = "robust"

# ──────────────────────────────────────────────────────────────────────
# Token GA guidance selection switch
# ──────────────────────────────────────────────────────────────────────
# Default to explanation to keep upstream batch experiments unchanged and avoid extra masking queries.
# Options: "explanation" | "random" | "masking"
DEFAULT_GUIDANCE_MODE = "explanation"
VALID_GUIDANCE_MODES = {"explanation", "random", "masking"}

from explain.coca_explainer import CocaExplainer
from explain.robust_explainer import RobustExplainer
from explain.mapping import map_explanation_to_source,ExplanationMapping
from src.model.wrapper import ModelWrapper
from src.utils.gen_embedding import read_json


# ══════════════════════════════════════════════════════════════════════
# Infrastructure
# ══════════════════════════════════════════════════════════════════════

class LineTracker:
    """Track line-number offsets across stages."""

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
    """Track variable renaming across stages for resolving dep_variable in the data-flow stage."""

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
    Track key variants throughout the attack process:
      best_variant:          variant with the lowest true_conf (largest confidence drop)
      first_success_variant: first variant that flips the prediction
      final_variant:         final variant
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
# Helper functions
# ══════════════════════════════════════════════════════════════════════

def _negate_simple_expr(expr):
    """Negate a simple expression."""
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
    """Convert input to str consistently."""
    if isinstance(code, bytes):
        return code.decode('utf-8')
    return code


def _query_from_lines(lines, wrapper, true_label, state):
    """Build an embedding from code lines, query the model, update state, and return (flipped, code string)."""
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
    Adaptive Re-Explanation triggering criterion.

    Args:
        original_true_conf: true-label confidence on the original code
        best_true_conf:     lowest true-label confidence reached by the token stage
        delta_low / delta_high: lower and upper thresholds (from module-level constants, can be overridden)

    Returns:
        (do_reexplain: bool, reason: str)
    """
    drop = original_true_conf - best_true_conf
    if drop < delta_low:
        return False, (
            f"drop={drop:.4f} < delta_low={delta_low:.2f}; the token stage barely changes the model; "
            f"re-explanation has low information gain, skipping"
        )
    if drop > delta_high:
        return False, (
            f"drop={drop:.4f} > delta_high={delta_high:.2f}; the token stage has substantially lowered confidence; "
            f"skipping parsing overhead and running structure transformation directly"
        )
    return True, f"drop={drop:.4f} in [{delta_low:.2f}, {delta_high:.2f}]; information-gain sweet spot"


def _identifier_lines(source_code, identifiers):
    """Return the line numbers where each identifier appears in the source code to normalize guidance formats."""
    source_text = source_code if isinstance(source_code, str) else source_code.decode('utf-8')
    lines = source_text.split('\n')
    line_map = {}
    for ident in identifiers:
        pattern = re.compile(r'\b' + re.escape(ident) + r'\b')
        line_map[ident] = [idx + 1 for idx, line in enumerate(lines) if pattern.search(line)]
    return line_map


def _random_identifier_importance(source_code, lang='c'):
    """Random guidance: randomize only identifier priorities without extra model queries."""
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
    Masking guidance: following masking.py, use the true-label
    confidence drop after replacing an identifier with MASK as its importance. Called only when guidance_mode="masking".
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
    """Generate a unified sorted list of [{'name', 'importance', 'lines'}] according to guidance_mode."""
    mode = (guidance_mode or DEFAULT_GUIDANCE_MODE).lower()
    if mode not in VALID_GUIDANCE_MODES:
        raise ValueError(
            f"guidance_mode must be one of {sorted(VALID_GUIDANCE_MODES)}, got {guidance_mode!r}"
        )

    if mode == "explanation":
        return mapping.get_identifier_importance(current_code_str, lang)
    if mode == "random":
        return _random_identifier_importance(current_code_str, lang)
    return _masking_identifier_importance(
        current_code_str, current_pdg, wrapper, wv, true_label, lang, verbose=verbose
    )


# ══════════════════════════════════════════════════════════════════════
# Stage 1: token-level attack
# ══════════════════════════════════════════════════════════════════════

def _attack_token(current_code_str, mapping, wrapper: ModelWrapper, wv, true_label, 
                  rename_map, state, lang, max_attempts):
    """
    Token-level attack: use Word2Vec to find candidates and quickly evaluate them by modifying PDG node attributes.
    """
    current_pdg = src2pdg(current_code_str)
    attempts = 0

    # 1. Initialize the environment: extract all replaceable identifiers from the code
    raw_ids = extract_identifiers_from_one_src(current_code_str.encode('utf-8'), lang)
    unique_identifiers = list(set(raw_ids))
    existing_identifiers = set(raw_ids)

    # 2. Iterate over key (vulnerable) nodes identified by the explainer
    for node in mapping.vulnerable_nodes:
        if attempts >= max_attempts:
            break

        line_no = node['line_no']
        lines = current_code_str.split('\n')
        line_idx = line_no - 1
        
        if line_idx < 0 or line_idx >= len(lines):
            continue

        line_content = lines[line_idx]

        # Filter replaceable identifiers contained in the current line
        line_targets = [
            v for v in unique_identifiers
            if re.search(r'\b' + re.escape(v) + r'\b', line_content)
        ]
        print(f"Targeting identifiers in line {line_no}: {line_targets}")

        # 3. Try replacements for each variable in the current line
        for target_var in line_targets:
            if attempts >= max_attempts:
                break

            # Generate W2V candidates and filter out existing identifiers
            candidates = gen_candis_w2v(target_var, wv, top_k=5)
            if not candidates:
                continue

            valid_candidates = [c for c in candidates if c not in existing_identifiers]
            if not valid_candidates:
                continue

            # 4. Try each candidate
            for id_candi in valid_candidates:
                if attempts >= max_attempts:
                    break

                # Use the fast path: directly modify PDG node features without regenerating CPG/PDG
                proposed_data = renamed_pdg_to_embedding(
                    current_pdg, wv, target_var, id_candi, true_label
                ).to(DEVICE)

                pred, true_conf = wrapper.predict_label_and_true_conf(
                    proposed_data, true_label
                )
                attempts += 1

                # Generate the source-code string after replacement
                proposed_code = _to_str(
                    rename_identifier(current_code_str, target_var, id_candi, lang)
                )
                
                # Update the global attack state
                state.update(proposed_code, pred, true_conf, true_label)

                # If the label is successfully flipped, return immediately
                if pred != true_label:
                    rename_map.add(target_var, id_candi)
                    return True, proposed_code

            # 5. Accumulated perturbation handling:
            # If no candidate in this round flips the label, keep the first candidate by default (greedy accumulation)
            # Note: this can be changed to "keep only the best candidate that lowers confidence" if desired
            best = valid_candidates[0]
            current_code_str = _to_str(
                rename_identifier(current_code_str, target_var, best, lang)
            )
            
            # After applying accumulated perturbations, reparse the PDG to ensure the next replacement uses the correct graph structure
            current_pdg = src2pdg(current_code_str)
            rename_map.add(target_var, best)

            # Update the identifier set of the current code
            existing_identifiers.discard(target_var)
            existing_identifiers.add(best)
            unique_identifiers = [best if v == target_var else v for v in unique_identifiers]

    return False, current_code_str

def _attack_token_drop(current_code_str, mapping, wrapper: ModelWrapper, wv, true_label,
                       rename_map, state, lang, max_attempts):
    """
    Token-level attack:
    Prefer replacements that significantly reduce the margin;
    If no candidate reduces the margin, keep the first valid candidate by default.

    margin = true_conf - other_conf

    The smaller the margin, the closer the attack is to success;
    When margin < 0, the other_label score exceeds the true_label score and the prediction flips.
    """
    current_pdg = src2pdg(current_code_str)
    attempts = 0

    # Get the initial margin as the baseline
    initial_data = pdg2embedding(current_pdg, wv, true_label).to(torch.device(DEVICE))

    _, current_baseline_conf, current_baseline_margin = \
        wrapper.predict_label_and_true_conf_margin(initial_data, true_label)

    raw_ids = extract_identifiers_from_one_src(
        current_code_str.encode('utf-8'), lang
    )
    unique_identifiers = list(set(raw_ids))
    existing_identifiers = set(raw_ids)

    # Cache tokenize results and recompute only when the code changes
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

            # Reuse tokenize results and retokenize only when the code changes
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

            # Record evaluated candidate results so the baseline can be updated during fallback
            evaluated_results = {}

            for id_candi in valid_candidates:
                if attempts >= max_attempts:
                    break

                # Fast evaluation: use renamed_pdg_to_embedding to avoid repeatedly reparsing the PDG
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

                # Keep the stored structure unchanged and still record true_conf
                state.update(proposed_code_tmp, pred, true_conf, true_label)

                evaluated_results[id_candi] = {
                    "pred": pred,
                    "true_conf": true_conf,
                    "margin": margin,
                }

                # If the label is successfully flipped, return immediately
                if pred != true_label:
                    rename_map.add(target_var, id_candi)
                    return True, proposed_code_tmp

                # If not flipped, find the candidate that reduces the margin the most
                if margin < min_margin_for_var:
                    min_margin_for_var = margin
                    best_true_conf_for_var = true_conf
                    best_candi_for_var = id_candi

            # If the query budget is exhausted and no candidate has been evaluated, do not force fallback
            if attempts >= max_attempts and not evaluated_results:
                break

            # Prefer the candidate that reduces the margin the most;
            # If no candidate reduces the margin, keep the first valid candidate by default.
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

                # If the fallback candidate has already been evaluated, update the baseline directly with its result
                if chosen_candi in evaluated_results:
                    current_baseline_margin = evaluated_results[chosen_candi]["margin"]
                    current_baseline_conf = evaluated_results[chosen_candi]["true_conf"]
                else:
                    # Rare case: chosen_candi was not evaluated, for example because max_attempts was reached midway
                    # Do not spend extra queries here; keep the old baseline
                    pass

            current_code_str = _to_str(
                rename_identifier(
                    current_code_str, target_var, chosen_candi, lang
                )
            )

            # Regenerate the PDG after applying accumulated perturbations
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
    Token-level attack based on a genetic algorithm.

    Chromosome encoding:
      - Each gene corresponds to a replaceable identifier
      - Gene value 0 = no replacement; 1..K = use the k-th W2V candidate

    Fitness:
      - margin = true_conf - other_conf, smaller is better
      - margin < 0 indicates prediction flipping (attack success)

    Guidance integration:
      - guidance_mode determines the identifier importance source: explanation / random / masking
      - identifier importance scores affect initialization and mutation probabilities
      - high-importance identifiers are more likely to be replaced and mutated more frequently
      - low-importance identifiers act as a fallback and can still participate in evolution

    Args:
        current_code_str:   current source-code string
        mapping:            ExplanationMapping object
        wrapper:            ModelWrapper instance
        wv:                 gensim Word2Vec vocabulary
        true_label:         ground-truth label
        rename_map:         RenameMap object (records renaming across stages)
        state:              AttackState object
        lang:               language identifier
        max_queries:        maximum query budget
        pop_size:           population size
        max_generations:    maximum number of generations
        top_k_candidates:   number of W2V candidates per identifier
        elite_count:        number of elites to keep
        crossover_rate:     crossover probability
        base_mutation_rate: base mutation probability
        batch_eval_size:    batch evaluation size (0 = evaluate one by one)
        verbose:            whether to print details
        guidance_mode:      identifier guidance source; default is explanation

    Returns:
        (success: bool, final_code: str)
    """
    import random as rnd

    current_pdg = src2pdg(current_code_str)

    # ═══════════════════════════════════════════════════════════
    # Step 1: Build the identifier table, candidate table, and importance weights
    # ═══════════════════════════════════════════════════════════

    guidance_mode = (guidance_mode or DEFAULT_GUIDANCE_MODE).lower()

    # Get the importance of each identifier according to guidance_mode.
    # explanation is the default path; random does not query the model; only masking performs extra masking-importance queries.
    ranked_identifiers = _get_guided_identifier_importance(
        guidance_mode, current_code_str, current_pdg,
        mapping, wrapper, wv, true_label, lang, verbose=verbose,
    )

    if not ranked_identifiers:
        if verbose:
            print("[GA] No replaceable identifiers")
        return False, current_code_str

    # Extract all identifiers already present in the current code (for conflict detection)
    existing_ids = set(item['name'] for item in ranked_identifiers)

    # Generate W2V candidates for each identifier and filter them
    identifiers = []     # identifier-name list
    candidates = []      # corresponding candidate list (excluding the original name)
    importances = []     # corresponding importance scores

    # Precompute: tokenize the same code only once and share the result across N variables
    _precomputed = precompute_tokenize(current_code_str)

    for item in ranked_identifiers:
        name = item['name']
        imp = item['importance']

        # candis = gen_candis_w2v(name, wv, top_k=top_k_candidates)
        candis = gen_candis(current_code_str, MLM, name, _precomputed=_precomputed)
        # candis = gen_candis_codet5(current_code_str, MLM, name, _precomputed=_precomputed)
        if not candis:
            continue

        # Filter out identifiers that already exist
        valid = [c for c in candis if c not in existing_ids]
        if not valid:
            continue

        identifiers.append(name)
        candidates.append(valid)
        importances.append(imp)

    num_genes = len(identifiers)
    if num_genes == 0:
        if verbose:
            print("[GA] No legal candidates for any identifier")
        return False, current_code_str

    # Normalize importance to [0, 1]
    max_imp = max(importances) if max(importances) > 0 else 1.0
    norm_importances = [imp / max_imp for imp in importances]

    if verbose:
        print(f"[GA] guidance={guidance_mode}, identifiers={num_genes}, "
              f"population={pop_size}, max_generations={max_generations}")
        top3 = [(identifiers[i], f"{importances[i]:.4f}") for i in range(min(3, num_genes))]
        print(f"[GA] Top-3 identifiers: {top3}")

    queries_used = 0

    def trace_replacements(chromosome):
        """Construct identifier-level trace information for actual replacements in the current chromosome."""
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
    # Step 2: Helper functions for chromosome operations
    # ═══════════════════════════════════════════════════════════

    def decode(chromosome):
        """Decode a chromosome into a replacement dictionary {old_name: new_name}."""
        rename_dict = {}
        for i, gene in enumerate(chromosome):
            if gene > 0 and gene <= len(candidates[i]):
                rename_dict[identifiers[i]] = candidates[i][gene - 1]
        return rename_dict

    # Add a cache to avoid reevaluating identical offspring
    fitness_cache = {}

    def evaluate(chromosome, generation=None, individual_index=None):
        """
        Evaluate the fitness of one chromosome.

        Returns:
            (margin, pred, true_conf, proposed_code_str)
            smaller margin is better; < 0 indicates a flip
        """
        key = tuple(chromosome)
        if key in fitness_cache:
            return fitness_cache[key]
        nonlocal queries_used

        rename_dict = decode(chromosome)
        if not rename_dict:
            return 999.0, true_label, 1.0, current_code_str

        # Use multi_renamed_pdg_to_embedding for fast evaluation
        proposed_data = multi_renamed_pdg_to_embedding(
            current_pdg, wv, rename_dict, true_label
        ).to(DEVICE)

        pred, true_conf, margin = wrapper.predict_label_and_true_conf_margin(
            proposed_data, true_label
        )
        queries_used += 1

        # Generate the corresponding source code (for logging and return)
        proposed_code = _to_str(
            rename_identifiers(current_code_str,rename_dict)
        )

        # Update the global attack state
        state.update(proposed_code, pred, true_conf, true_label)

        # Record replacements and explanation scores for each actual model query for later vulnerable-space analysis.
        # When tracing is disabled by default, do not build replacements to avoid extra overhead in normal batch attacks.
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
        Generate an initial chromosome.
        High-importance identifiers are more likely to be initialized in a replaced state.
        """
        chromosome = [0] * num_genes
        for i in range(num_genes):
            # Initial replacement probability = 0.2 + 0.7 * normalized_importance
            # importance=1.0 → 90% probability of being replaced
            # importance=0.0 → 20% probability of being replaced
            p_init = 0.2 + 0.7 * norm_importances[i]
            if rnd.random() < p_init:
                chromosome[i] = rnd.randint(1, len(candidates[i]))
        return chromosome

    def mutate(chromosome):
        """
        Mutation operation: mutation probability weighted by importance.
        """
        result = list(chromosome)
        for i in range(num_genes):
            # Mutation probability = base_rate * (1 + importance)
            p_mut = base_mutation_rate * (1.0 + norm_importances[i])
            if rnd.random() < p_mut:
                result[i] = rnd.randint(0, len(candidates[i]))
        return result

    def crossover(parent_a, parent_b):
        """Uniform crossover: choose each gene independently from two parents."""
        child = [0] * num_genes
        for i in range(num_genes):
            if rnd.random() < 0.5:
                child[i] = parent_a[i]
            else:
                child[i] = parent_b[i]
        return child

    def tournament_select(population, fitnesses, k=3):
        """Tournament selection: randomly sample k individuals and choose the best."""
        indices = rnd.sample(range(len(population)), min(k, len(population)))
        best_idx = min(indices, key=lambda x: fitnesses[x])
        return population[best_idx]

    # ═══════════════════════════════════════════════════════════
    # Step 3: Main GA loop
    # ═══════════════════════════════════════════════════════════

    # Initialize the population
    population = [init_chromosome() for _ in range(pop_size)]

    best_ever_margin = 999.0
    best_ever_chromosome = None
    best_ever_code = current_code_str

    for gen in range(max_generations):
        if queries_used >= max_queries:
            break

        # ── Evaluate the current population ──
        fitnesses = []
        for idx, chrom in enumerate(population):
            if queries_used >= max_queries:
                fitnesses.append(999.0)
                continue

            margin, pred, true_conf, proposed_code = evaluate(
                chrom, generation=gen, individual_index=idx
            )
            fitnesses.append(margin)

            # Update the global best
            if margin < best_ever_margin:
                best_ever_margin = margin
                best_ever_chromosome = list(chrom)
                best_ever_code = proposed_code

            # Attack succeeds: return immediately
            if pred != true_label:
                if verbose:
                    rename_dict = decode(chrom)
                    renames = ', '.join(f'{k}→{v}' for k, v in rename_dict.items())
                    print(f"[GA] ✓ Generation {gen}, individual {idx} flipped successfully!"
                          f" margin={margin:.4f} queries={queries_used}")
                    print(f"[GA]   replacements: {renames}")

                # Record renaming in rename_map
                for old_n, new_n in decode(chrom).items():
                    rename_map.add(old_n, new_n)

                return True, proposed_code

        if queries_used >= max_queries:
            break

        # ── Print current-generation statistics ──
        if verbose:
            gen_best = min(fitnesses)
            gen_avg = sum(f for f in fitnesses if f < 999) / max(1, sum(1 for f in fitnesses if f < 999))
            num_replacing = sum(1 for g in population[fitnesses.index(gen_best)] if g > 0)
            print(f"[GA] Generation {gen}: best_margin={gen_best:.4f} "
                  f"avg={gen_avg:.4f} replacements={num_replacing} "
                  f"queries={queries_used}/{max_queries}")

        # ── Build the next generation ──
        new_population = []

        # Elite preservation
        sorted_indices = sorted(range(len(fitnesses)), key=lambda x: fitnesses[x])
        for i in range(min(elite_count, len(sorted_indices))):
            new_population.append(list(population[sorted_indices[i]]))

        # Generate the remaining individuals via crossover + mutation
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

        # ── Early stopping: best margin does not improve for 3 consecutive generations ──
        if gen >= 3:
            # Simple stagnation check
            pass  # Optional: record historical best and break after consecutive non-improvements

    # ═══════════════════════════════════════════════════════════
    # Step 4: GA ends; return the code corresponding to the best individual
    # ═══════════════════════════════════════════════════════════

    if best_ever_chromosome is not None:
        rename_dict = decode(best_ever_chromosome)
        for old_n, new_n in rename_dict.items():
            rename_map.add(old_n, new_n)

        if verbose:
            num_replaced = sum(1 for g in best_ever_chromosome if g > 0)
            print(f"[GA] ✗ Not flipped, best_margin={best_ever_margin:.4f} "
                  f"replacements={num_replaced} total_queries={queries_used}")

    return False, best_ever_code



# ══════════════════════════════════════════════════════════════════════
# Result construction
# ══════════════════════════════════════════════════════════════════════

def _build_attack_result(sample_id, true_label, original_code, state,
                         original_pred, original_true_conf, wrapper):
    """Collect information from AttackState to build an AttackResult."""
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
# Main entry point
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
    # ── Ablation switch (default "full" = token→reexplain→struct, matching the main paper pipeline) ──
    mode: str = DEFAULT_MODE,
    reexplain_fn=None,
    # Adaptive Re-Explanation switch
    adaptive_reexplain: bool = True,
    # query-level attack trace output directory; None uses the default directory in attack_trace.py
    trace_dir=None,
):
    """
    ExplAtk: explainer-guided iterative adversarial attack.

    Args:
        sample_id:                sample ID
        original_code:            original source code (str or bytes)
        true_label:               ground-truth label
        mapping:                  ExplanationMapping object (for original_code)
        wrapper:                  ModelWrapper instance
        wv:                       gensim Word2Vec vocabulary
        lang:                     language identifier (default: 'c')
        max_token_attempts:       maximum number of token-stage attempts
        max_dependency_attempts:  maximum number of structure-transformation-stage attempts
        verbose:                  whether to print progress

        mode: ablation mode, one of three options:
          - "token_only"  : Explain(v0) → Token        (run only the token stage)
          - "struct_only" : Explain(v0) → structure transformation (run only structure transformation)
          - "full"        : Explain(v0) → Token → Re-Explain(v1) → structure transformation
        reexplain_fn: callable(code_str) -> ExplanationMapping
                      called once only when mode="full"; regenerates the mapping for the
                      variant produced by the token stage. Falls back to the old mapping if it fails or is not provided.
        adaptive_reexplain: whether to enable Adaptive Re-Explanation when mode="full".
                            True: adaptively decide whether to actually call reexplain_fn based on conf_drop
                                  (see _should_reexplain and module-level constants DELTA_LOW/HIGH);
                            False: always re-explain (for ablation comparison).

    Returns:
        AttackResult
    """
    assert mode in ("token_only", "struct_only", "full"), \
        f"mode must be 'token_only' / 'struct_only' / 'full', got {mode!r}"

    wrapper.reset_query_count()
    ori_code_str = original_code if isinstance(original_code, str) else original_code.decode('utf-8')

    # ── Original prediction ──
    from common.utils.gen_embedding import src2embedding
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_data = src2embedding(ori_code_str.encode('utf-8'), true_label).to(DEVICE)
    original_pred, original_true_conf = wrapper.predict_label_and_true_conf(
        original_data, true_label
    )

    if original_pred != true_label:
        if verbose:
            print(f"[ExplAtk] Sample {sample_id}: prediction {original_pred} != true label {true_label}; skipping")
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
        print(f"[ExplAtk] mode: {mode} (token={run_token}, "
              f"reexplain={run_reexplain}, struct={run_struct})")

    # ════════════════════════════════════════════════════════════
    # Stage 1: GA token-level attack
    # ════════════════════════════════════════════════════════════
    if run_token:
        if verbose:
            print(f"[ExplAtk] Stage 1: GA token attack (budget={max_token_attempts})")

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
                print(f"  ✓ Token attack succeeded! {renames}  queries: {wrapper.get_query_count()}")
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
            print(f"  ✗ Token attack did not succeed (queries: {wrapper.get_query_count()})")

        # ── Fall back to the lowest-confidence variant from the token stage ──
        if (state.best_variant is not None
                and state.best_true_conf < state.final_true_conf):
            current_code = state.best_variant
            if verbose:
                print(f"  ↩ Falling back to the best variant (conf={state.best_true_conf:.4f}"
                      f" < final={state.final_true_conf:.4f})")

    # token_only mode: stop here
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
    # Intermediate step: run re-explain once between token and structure transformation
    # Triggered only in mode="full"; struct_only skips it and keeps the v0 mapping.
    # When adaptive_reexplain=True, use conf_drop to adaptively decide whether to run it.
    # ════════════════════════════════════════════════════════════
    if run_reexplain:
        # —— Adaptive early-stop decision —————————————————————————————
        skip_by_adaptive = False
        if adaptive_reexplain:
            do_re, reason = _should_reexplain(
                state.original_true_conf, state.best_true_conf,
            )
            if verbose:
                tag = "run" if do_re else "skip"
                print(f"[ExplAtk] Adaptive Re-Explain → {tag}: {reason}")
            skip_by_adaptive = not do_re

        if skip_by_adaptive:
            pass  # Do not call reexplain_fn; keep the old mapping
        elif reexplain_fn is None:
            if verbose:
                print("[ExplAtk] Re-Explain skipped: reexplain_fn is not provided; keeping the old mapping")
        else:
            if verbose:
                print("[ExplAtk] Re-Explain: re-explaining the variant produced by the token stage")
            try:
                new_mapping = reexplain_fn(current_code)
                if new_mapping is not None:
                    current_mapping = new_mapping
                    if verbose:
                        print(f"  ✓ Refreshed mapping "
                              f"(key_nodes={len(current_mapping.vulnerable_nodes)})")
                else:
                    if verbose:
                        print("  ✗ reexplain_fn returned None; keeping the old mapping")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Re-Explain failed ({e}); keeping the old mapping")

    # ════════════════════════════════════════════════════════════
    # Stage 2: structure transformation attack guided by DDG/CDG edges
    # ════════════════════════════════════════════════════════════
    if verbose:
        print(f"[ExplAtk] Stage 2: structure transformation attack")

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
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {status}  total_queries: {wrapper.get_query_count()}")

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
        print("First successful variant:")
        print(f"{'─' * 60}")
        print(result.first_success_variant)


def replace_path_part(path, old_part, new_part):
    """
    Replace only one directory component in a path to avoid accidentally replacing strings in the filename.
    """
    path = Path(path)
    parts = list(path.parts)

    try:
        idx = parts.index(old_part)
    except ValueError:
        raise ValueError(f"Directory component {old_part} not found in path: {path}")

    parts[idx] = new_part
    return Path(*parts)


def source_to_related_paths(source_path):
    """
    Automatically infer normal / ori from source_path and derive:
        dot_path
        cpg_bin_path
        json_path

    Rules:
    1. normal:
        normal-src -> normal-pdg
        normal-src -> normal-cpg-bin
        normal-src -> normal-embedding

    2. ori:
        BigVul/all-src -> BigVul/ori-pdg
        BigVul/all-src -> BigVul/ori-cpg-bin
        BigVul/all-src -> BigVul/ori-embedding

        other directories/src -> ori-pdg
        other directories/src -> ori-cpg-bin
        other directories/src -> ori-embedding

    3. Suffixes:
        .c -> .dot
        .c -> .bin
        .c -> .json
    """

    source_path = Path(source_path)

    if source_path.suffix != ".c":
        raise ValueError(f"Only .c files are currently supported, got: {source_path}")

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
            f"Cannot infer the path type automatically; the path should contain normal-src, BigVul/all-src, or src: {source_path}"
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
    # ── Ablation switch: forwarded to expl_atk ──
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
    pred_label = wrapper.predict_label(data)  # usually 1
    result = explainer.explain(data, pred_label)

    # 2. Map back to source code (full information)
    mapping = map_explanation_to_source(
        explain_result=result,
        dot_path=dot_path,
        cpg_bin_path=cpg_bin_path,
        source_path=source_path,     # optional
    )

    mapping.print_summary()

    # ── Build the re-explain closure: called once by expl_atk only when mode="full" ──
    # The closure captures explainer / pred_label and regenerates dot+cpg-bin locally for new code.
    def _reexplain_fn(code_str: str):
        """
        Regenerate PDG/CPG-bin for the current variant, then run the explainer again,
        and return a new ExplanationMapping. Any intermediate failure raises an exception,
        which is handled by the try/except inside expl_atk by falling back to the old mapping.
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

            # Generate cpg-bin and dot
            joern_parse(tmp_src, tmp_bin)
            if not os.path.exists(tmp_bin):
                raise RuntimeError("joern_parse did not produce cpg-bin")
            joern_export(tmp_bin, tmp_pdg_dir)
            if not os.path.exists(tmp_dot):
                raise RuntimeError("joern_export did not produce dot")

            # Run the explainer using an embedding based on the variant code
            new_data = src2embedding(
                code_str.encode("utf-8") if isinstance(code_str, str) else code_str,
                pred_label,
            ).to(DEVICE)
            new_result = explainer.explain(new_data, pred_label)

            new_mapping = map_explanation_to_source(
                explain_result=new_result,
                dot_path=tmp_dot,
                cpg_bin_path=tmp_bin,
                source_path=None,  # Use variant source code instead of the original source_path, which is no longer accurate
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