import random
import sys
from pathlib import Path

_METHOD_ROOT = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_METHOD_ROOT, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
from common.attack_result import AttackResult, to_text
from common.config_loader import get_settings
from common.utils.gen_embedding import (
    load_word_vectors,
    multi_renamed_pdg_to_embedding,
    pdg2embedding,
    renamed_pdg_to_embedding,
    src2pdg,
)
from common.utils.renamer import rename_identifier, rename_identifiers
from src.attack.masking import importance_sampling
from src.model.wrapper import ModelWrapper
from src.utils.gen_candidates import gen_candis, init_mlm, gen_candis_w2v

S = get_settings()
DEVICE = S.device
MHM_MAX_ITER = S.mhm_max_iter
MODEL_NAME = "reveal"
CHECKPOINT_PATH = S.reveal_checkpoint
POP_SIZE = S.pop_size
MAX_GEN = S.max_gen
MUTATION_RATE = S.mutation_rate

# ================= ALERT resource-control parameters =================
# Use these defaults if the corresponding fields are missing from settings.
ALERT_TOP_N_VARS = int(getattr(S, "alert_top_n_vars", 20))
ALERT_TOP_K_CANDIS = int(getattr(S, "alert_top_k_candis", 8))

# Maximum number of queries per sample. <= 0 means no limit.
ALERT_MAX_QUERIES = int(getattr(S, "alert_max_queries", 500))

# Stop GA early after this many consecutive generations without improvement.
ALERT_GA_PATIENCE = int(getattr(S, "alert_ga_patience", 5))

# For full-scale runs, alert_verbose=False is recommended.
ALERT_VERBOSE = bool(getattr(S, "alert_verbose", False))
ALERT_PROGRESS = bool(getattr(S, "alert_progress", True))


def _vprint(*args, **kwargs):
    """Detailed candidate-level / GA-individual-level logs. Disabled by default for full runs."""
    if ALERT_VERBOSE:
        print(*args, **kwargs)


def _pprint(*args, **kwargs):
    """Sample-level / success/failure progress logs."""
    if ALERT_PROGRESS:
        print(*args, **kwargs)


def _budget_exceeded(wrapper):
    """Per-sample query budget control."""
    return ALERT_MAX_QUERIES > 0 and wrapper.get_query_count() >= ALERT_MAX_QUERIES

def _build_result(
    sample_id,
    true_label,
    original_code,
    final_variant,
    best_variant,
    first_success_variant,
    original_pred,
    original_true_conf,
    final_pred,
    final_true_conf,
    best_true_conf,
    success_true_conf,
    wrapper: ModelWrapper,
):
    is_attackable = original_pred == true_label
    success = is_attackable and final_pred != true_label
    return AttackResult(
        sample_id=sample_id,
        attack_name="alert",
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
        first_success_variant=to_text(first_success_variant) if first_success_variant is not None else None,
        final_pred=final_pred,
        final_true_conf=final_true_conf,
        best_true_conf=best_true_conf,
        success_true_conf=success_true_conf,
    )


def alert_attack(src, true_label, wrapper: ModelWrapper, lang='cpp', max_iter=MHM_MAX_ITER, sample_id="unknown"):
    torch.cuda.empty_cache()
    if isinstance(src, str):
        current_code = src.encode('utf-8')
    else:
        current_code = src

    wrapper.reset_query_count()
    ori_src_bytes = current_code
    original_code = to_text(current_code)

    # extract identifier & importance sampling
    sampled_var_list = importance_sampling(current_code, true_label, wrapper)
    candis_dict = {}
    mlm_model = init_mlm()

    # Keep only the top-N important variables to reduce the later greedy-search space.
    if ALERT_TOP_N_VARS > 0:
        sampled_var_list = sampled_var_list[:ALERT_TOP_N_VARS]

    sampled_vars = [item[0] for item in sampled_var_list]
    greedy_chromosome = {v: v for v in sampled_vars}

    wv = load_word_vectors()

    # Alert Iter
    # current_data = src2embedding(current_code, true_label).to(torch.device(DEVICE))
    # ================= Stage 1: Greedy Search =================
    current_pdg = src2pdg(current_code)
    ori_pdg = current_pdg
    current_data = pdg2embedding(current_pdg, wv, true_label).to(torch.device(DEVICE))
    original_pred, original_true_conf = wrapper.predict_label_and_true_conf(current_data, true_label)
    best_variant = current_code
    best_true_conf = original_true_conf
    first_success_variant = None
    success_true_conf = None
    if original_pred != true_label:
        return _build_result(
            sample_id,
            true_label,
            original_code,
            current_code,
            best_variant,
            first_success_variant,
            original_pred,
            original_true_conf,
            original_pred,
            original_true_conf,
            best_true_conf,
            success_true_conf,
            wrapper,
        )
    for index, target_var in enumerate(sampled_vars):
        if _budget_exceeded(wrapper):
            _pprint(f"[Budget] Stop greedy search: query_count={wrapper.get_query_count()}")
            break

        _vprint('-' * 65)

        if target_var not in candis_dict:
            candis = gen_candis(current_code, mlm_model, target_var)
            # candis = gen_candis_w2v(target_var, wv, 5)

            # Deduplicate candidates and limit each variable to at most top-K candidates.
            candis = list(dict.fromkeys(candis))
            if ALERT_TOP_K_CANDIS > 0:
                candis = candis[:ALERT_TOP_K_CANDIS]

            candis_dict[target_var] = candis
            _vprint(f"Gen '{target_var}' Candis: '{candis_dict[target_var]}'")

        id_candis = candis_dict[target_var]
        if not id_candis:
            _vprint(f"Iter {index + 1}: no candidates for '{target_var}'")
            continue

        ori_prob = wrapper.predict_prob(current_data, true_label)

        best_score = 0.0
        best_candi = None
        best_proposed_code = None

        for k, id_candi in enumerate(id_candis):
            if _budget_exceeded(wrapper):
                _pprint(f"[Budget] Stop candidate search: query_count={wrapper.get_query_count()}")
                break

            # Candidate identifiers must not have the same names as existing identifiers.
            if id_candi in sampled_vars:
                _vprint(
                    f"Iter {index + 1} Attempt {k + 1}: Failed! "
                    f"'{id_candi}' has the same name with other ids!"
                )
                continue

            _vprint(
                f"Iter {index + 1} Attempt {k + 1}: "
                f"Trying to rename '{target_var}' to '{id_candi}'"
            )

            # Rename the identifier in the source code
            proposed_code = rename_identifier(current_code, target_var, id_candi, lang).encode('utf-8')

            # Candidate Data is used only for this prediction and is not saved long term.
            proposed_data = renamed_pdg_to_embedding(
                current_pdg, wv, target_var, id_candi, true_label
            ).to(torch.device(DEVICE))

            pred_label, proposed_true_conf = wrapper.predict_label_and_true_conf(
                proposed_data, true_label
            )

            if proposed_true_conf < best_true_conf:
                best_true_conf = proposed_true_conf
                best_variant = proposed_code

            if first_success_variant is None and pred_label != true_label:
                first_success_variant = proposed_code
                success_true_conf = proposed_true_conf

            if pred_label != true_label:
                # Release the candidate GPU Data reference before returning.
                del proposed_data

                return _build_result(
                    sample_id,
                    true_label,
                    original_code,
                    proposed_code,
                    best_variant,
                    first_success_variant,
                    original_pred,
                    original_true_conf,
                    pred_label,
                    proposed_true_conf,
                    best_true_conf,
                    success_true_conf,
                    wrapper,
                )

            # Previously this used wrapper.compute_importance(true_label, ori_prob, proposed_data).
            # However, compute_importance internally performs another forward pass, so the same candidate would be predicted twice.
            # Reuse proposed_true_conf directly here.
            score = max(0.0, ori_prob - proposed_true_conf)

            if score > best_score:
                best_score = score
                best_candi = id_candi
                best_proposed_code = proposed_code

            # Release the candidate GPU Data reference immediately after candidate evaluation.
            del proposed_data

        if best_proposed_code is not None and best_candi is not None:
            current_code = best_proposed_code

            # Do not save best_proposed_data.
            # When this candidate is actually accepted, regenerate current_data to avoid keeping candidate Data on the GPU for too long.
            try:
                del current_data
            except UnboundLocalError:
                pass

            current_data = renamed_pdg_to_embedding(
                current_pdg, wv, target_var, best_candi, true_label
            ).to(torch.device(DEVICE))

            # Update the PDG corresponding to the current code for use by the next variable.
            current_pdg = src2pdg(current_code)
            greedy_chromosome[target_var] = best_candi

            _vprint(f"Greedy: '{target_var}' -> '{best_candi}' (Score: {best_score:.4f})")
        else:
            _vprint(f"Iter {index + 1} didn't find a better replacement for attack!")

    # ================= Stage 2: Genetic Algorithm =================
    
    # ================= Stage 2: Genetic Algorithm =================
    eff_pop_size = max(2, POP_SIZE)
    eff_max_gen = max(0, MAX_GEN)

    # Record the current best chromosome. Even if GA is skipped because of the budget, it can fall back to greedy_chromosome.
    best_ga_chromosome = greedy_chromosome.copy()

    # 1. Initialize the population.
    # Must include the best result produced by the greedy algorithm (elitism).
    population = [greedy_chromosome.copy()]

    # Fill the remaining population with random individuals.
    for _ in range(eff_pop_size - 1):
        random_chrom = {v: v for v in sampled_vars}
        for v in sampled_vars:
            if v in candis_dict and candis_dict[v]:
                random_chrom[v] = random.choice(candis_dict[v])
        population.append(random_chrom)

    # 2. Evolution loop: keep the original MAX_GEN but add early stopping.
    stale_gen = 0

    for gen in range(eff_max_gen):
        if _budget_exceeded(wrapper):
            _pprint(f"[Budget] Stop GA: query_count={wrapper.get_query_count()}")
            break

        _vprint(f"--- GA Generation {gen + 1}/{eff_max_gen} ---")
        pop_scores = []
        before_gen_best = best_true_conf

        # Compute fitness for the current population.
        for i, chrom in enumerate(population):
            if _budget_exceeded(wrapper):
                _pprint(f"[Budget] Stop GA population eval: query_count={wrapper.get_query_count()}")
                break

            fit, adv_code, is_succ, pred_label, adv_true_conf = compute_fitness(
                chrom, ori_src_bytes, ori_pdg, wv, true_label, wrapper, lang
            )

            _vprint(f"Population {i + 1}/{eff_pop_size}: fit({fit:.6f}) chrom({chrom})")

            if adv_true_conf < best_true_conf:
                best_true_conf = adv_true_conf
                best_variant = adv_code
                best_ga_chromosome = chrom.copy()

            if first_success_variant is None and is_succ:
                first_success_variant = adv_code
                success_true_conf = adv_true_conf

            if is_succ:
                _pprint(f"GA Success at Gen {gen + 1}, Individual {i + 1}")

                return _build_result(
                    sample_id,
                    true_label,
                    original_code,
                    adv_code,
                    best_variant,
                    first_success_variant,
                    original_pred,
                    original_true_conf,
                    pred_label,
                    adv_true_conf,
                    best_true_conf,
                    success_true_conf,
                    wrapper,
                )

            pop_scores.append((fit, chrom))

        # If no individual is evaluated because of the query budget or other reasons, exit GA directly.
        if not pop_scores:
            break

        # Sort in descending order by score.
        pop_scores.sort(key=lambda x: x[0], reverse=True)

        # Even if this generation does not update best_true_conf, record the current best-fitness chromosome as a fallback.
        best_ga_chromosome = pop_scores[0][1].copy()

        # GA early stopping: if this generation does not lower true_conf, increase the stale counter.
        if best_true_conf < before_gen_best:
            stale_gen = 0
        else:
            stale_gen += 1
            if stale_gen >= ALERT_GA_PATIENCE:
                _vprint(f"GA early stop: no improvement for {ALERT_GA_PATIENCE} generations")
                break

        # Elitism: keep the top 2 individuals directly in the next generation.
        new_pop = [x[1].copy() for x in pop_scores[:2]]

        # Breed the next generation.
        while len(new_pop) < eff_pop_size:
            if len(pop_scores) >= 2:
                parent1 = max(random.sample(pop_scores, 2), key=lambda x: x[0])[1]
                parent2 = max(random.sample(pop_scores, 2), key=lambda x: x[0])[1]
            else:
                parent1 = pop_scores[0][1]
                parent2 = pop_scores[0][1]

            child = crossover(parent1, parent2)
            child = mutate(child, candis_dict)

            new_pop.append(child)

        population = new_pop

    # GA also failed; return the greedy result or the best result found during GA.
    final_best_code = apply_chromosome(ori_src_bytes, best_ga_chromosome, lang)
    final_data = multi_renamed_pdg_to_embedding(
        ori_pdg, wv, best_ga_chromosome, true_label
    ).to(torch.device(DEVICE))
    final_pred, final_true_conf = wrapper.predict_label_and_true_conf(final_data, true_label)
    del final_data
    if final_true_conf < best_true_conf:
        best_true_conf = final_true_conf
        best_variant = final_best_code
    return _build_result(
        sample_id,
        true_label,
        original_code,
        final_best_code,
        best_variant,
        first_success_variant,
        original_pred,
        original_true_conf,
        final_pred,
        final_true_conf,
        best_true_conf,
        success_true_conf,
        wrapper,
    )


def run_alert_attack(
    model_name,
    checkpoint_path,
    source_code,
    true_label,
    sample_id="unknown",
    sample_i=None,
    lang='cpp',
    max_iter=MHM_MAX_ITER,
    input_dim=100,
    output_dim=200,
):
    effective_sample_id = sample_id if sample_id != "unknown" else (
        sample_i if sample_i is not None else "unknown"
    )
    wrapper = ModelWrapper(model_name, checkpoint_path, input_dim=input_dim, output_dim=output_dim)
    return alert_attack(
        src=source_code,
        true_label=true_label,
        wrapper=wrapper,
        lang=lang,
        max_iter=max_iter,
        sample_id=effective_sample_id,
    )
                
def apply_chromosome(code, chromosome, lang='cpp'):
    """
    Apply the replacement mapping to the code.
    chromosome: dict, e.g., {'sushu': 'cnt', 'res': 'ans'}
    """
    # Replace all variables at once based on their positions.
    temp_code = rename_identifiers(code, chromosome, lang)
    return temp_code

def compute_fitness(chromosome, original_code, ori_pdg, word_vectors, true_label, wrapper, lang):
    """
    Compute the fitness of a chromosome.
    Note: adv_data is used only for this prediction and its reference is released immediately after use.
    """
    adv_code = apply_chromosome(original_code, chromosome, lang)

    adv_data = multi_renamed_pdg_to_embedding(
        ori_pdg, word_vectors, chromosome, true_label
    ).to(torch.device(DEVICE))

    pred_label, prob = wrapper.predict_label_and_true_conf(adv_data, true_label)
    is_success = pred_label != true_label

    fitness = 1.0 - prob

    del adv_data

    return fitness, adv_code, is_success, pred_label, prob

def crossover(p1, p2):
    """Crossover: randomly select genes from the two parent dictionaries."""
    child = {}
    all_vars = set(p1.keys()) | set(p2.keys())
    for var in all_vars:
        # 50% chance to inherit from p1, 50% from p2.
        if random.random() < 0.5:
            child[var] = p1.get(var, var) # If p1 does not contain this key, keep the original name.
        else:
            child[var] = p2.get(var, var)
    return child

def mutate(chromosome, candis_dict):
    """Mutation: randomly change the mapping for one variable."""
    mutated = chromosome.copy()
    for var in mutated.keys():
        if random.random() < MUTATION_RATE:
            # Randomly select one candidate from this variable's candidate pool.
            if var in candis_dict and candis_dict[var]:
                mutated[var] = random.choice(candis_dict[var])
    return mutated

def test():
    ori_code = b'''
                hb_face_t * hb_face_create ( hb_blob_t * blob , unsigned int index ) {
                    hb_face_t * face ;
                    if ( unlikely ( ! blob || ! hb_blob_get_length ( blob ) ) ) return hb_face_get_empty ( ) ;
                    hb_face_for_data_closure_t * closure = _hb_face_for_data_closure_create ( OT : : Sanitizer < OT : : OpenTypeFontFile > : : sanitize ( hb_blob_reference ( blob ) ) , index ) ;
                    if ( unlikely ( ! closure ) ) return hb_face_get_empty ( ) ;
                    face = hb_face_create_for_tables ( _hb_face_for_data_reference_table , closure , ( hb_destroy_func_t ) _hb_face_for_data_closure_destroy ) ;
                    hb_face_set_index ( face , index ) ;
                    return face ;
                }
                '''
    
    normal_code = b'''
int FUN1(char* VAR1) {
    char VAR2[32];
    char VAR3[] = "";
    int VAR4 = 0;
    int VAR5;


    VAR5 = strlen(VAR1);
    FUN2("", VAR5);


    if (VAR5 > 0) {
        VAR4 = 1;
    }


    strcpy(VAR2, VAR1);


    FUN2("", VAR2);


    if (strlen(VAR2) > 0) {
        FUN2("");
        return VAR4;
    }

    return -1;
}                                                                                                             
'''
    result = run_alert_attack(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        source_code=ori_code,
        true_label=1,
        sample_id="demo_alert",
        lang='cpp',
        max_iter=10,
        input_dim=100,
        output_dim=200,
    )
    print(result.to_dict())


if __name__ == "__main__":
    test()