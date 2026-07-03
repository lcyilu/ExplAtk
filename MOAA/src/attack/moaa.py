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
import torch.nn.functional as F
from common.attack_result import AttackResult, to_text
from common.config_loader import get_settings
from common.utils.gen_embedding import src2embedding, src2pdg, pdg2embedding, load_word_vectors, renamed_pdg_to_embedding, multi_renamed_pdg_to_embedding
from common.utils.parser import extract_identifiers_from_one_src
from common.utils.renamer import rename_identifier, rename_identifiers
from src.attack.masking import importance_sampling
from src.utils.gen_candidates import gen_candis_codet5, gen_candis_w2v, init_mlm
from src.model.wrapper import ModelWrapper

S = get_settings()
DEVICE = S.device
MODEL_NAME = "reveal"
CHECKPOINT_PATH = S.reveal_checkpoint

# MOAA-specific (paper defaults; not tied to global pop_size / max_gen)
POP_SIZE = 50        # Population size
MOAA_MAX_GEN = 20    # Maximum number of generations
MUTATION_RATE = 0.4  # Mutation rate
CROSSOVER_RATE = 0.8 # Crossover rate
POST_SUCCESS_GENERATIONS = 2


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
        attack_name="moaa",
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


class Individual:
    """
    Individual class: stores the chromosome, objective values, rank, and crowding distance.
    """
    def __init__(self, chromosome):
        self.chromosome = chromosome  # Dict: {var_name: new_name}
        # Objectives: [Attack Score (Prob), Semantic Dist, Mod Rate]
        # Lower is better for all objectives.
        self.objectives = [None, None, None] 
        self.rank = 0
        self.crowding_distance = 0
        self.adv_code = None
        self.is_success = False
        self.pred_label = None
        self.true_conf = None


def moaa_attack(src, true_label, wrapper: ModelWrapper, lang='cpp', max_iter=MOAA_MAX_GEN, sample_id="unknown"):
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
    mlm_model = init_mlm() # The MLM model used by MOAA is CodeT5, while ALERT uses CodeBERT.

    sampled_vars = [item[0] for item in sampled_var_list]
    # Build a mapping from variable names to importance scores for the mutation strategy.
    importance_map = {item[0]: item[1] for item in sampled_var_list}

    w2v_model = load_word_vectors()

    # Alert Iter
    # current_data = src2embedding(current_code, true_label).to(torch.device(DEVICE))
    # ================= Phase 1: Greedy Search =================
    current_pdg = src2pdg(current_code)
    ori_pdg = current_pdg
    ori_data = pdg2embedding(current_pdg, w2v_model, true_label).to(torch.device(DEVICE))
    original_pred, original_true_conf = wrapper.predict_label_and_true_conf(ori_data, true_label)
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
        print('-'*65) # Iteration separator.
        if target_var not in candis_dict.keys():
            candis = gen_candis_codet5(current_code, mlm_model, target_var)
            # candis = gen_candis_w2v(w2v_model,target_var,10)
            id_candis = list(dict.fromkeys(candis))
            id_candis = [c for c in id_candis if c not in sampled_vars]
            if len(id_candis) == 0:
                print(f"No candidates generated for variable '{target_var}'. Skipping...")
                continue
            candis_dict[target_var] = id_candis
            print(f"Gen '{target_var}' Candis: '{candis_dict[target_var]}'")
     # ================= NSGA-II initialization =================
    
    population = []
    successful_archives = [] # Archive all successful attacks.
    first_success_gen = None

    # Individual 0: original code without modification.
    orig_chrom = {v: v for v in sampled_vars}
    population.append(Individual(orig_chrom))

    # Randomly generate the remaining individuals.
    # Note: mutual-exclusion rules must also be followed when generating the initial population.
    for _ in range(POP_SIZE - 1):
        chrom = {v: v for v in sampled_vars}
        used_names = set(sampled_vars) # Names occupied initially.

        for v in sampled_vars:
            if v in candis_dict and random.random() < 0.3:
                # Try to find a non-conflicting candidate.
                candidates = candis_dict[v]
                random.shuffle(candidates)
                for c in candidates:
                    # Ensure this name has not been used by other variables in the chromosome, even though candidates are unique.
                    if c not in used_names:
                        used_names.remove(chrom[v]) # Release the old name.
                        chrom[v] = c
                        used_names.add(c)
                        break
        population.append(Individual(chrom))

    # ================= Evolution loop =================
    for gen in range(max_iter):
        print(f"=== MOAA Generation {gen+1}/{max_iter} ===")
        
        # 1. Evaluation.
        for ind in population:
            if ind.objectives[0] is None:
                evaluate_objectives(ind, ori_src_bytes, ori_pdg, ori_data, w2v_model, true_label, wrapper, lang, sampled_vars)
                if ind.true_conf < best_true_conf:
                    best_true_conf = ind.true_conf
                    best_variant = ind.adv_code
                if ind.is_success:
                    if first_success_gen is None:
                        first_success_gen = gen
                    if first_success_variant is None:
                        first_success_variant = ind.adv_code
                        success_true_conf = ind.true_conf
                    successful_archives.append(ind)

        # 2. Generate offspring.
        offspring = []
        while len(offspring) < POP_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover + mutation with conflict checking.
            child_chrom = crossover(parent1.chromosome, parent2.chromosome)
            child_chrom = mutate(child_chrom, candis_dict, importance_map)
            
            offspring.append(Individual(child_chrom))
            
        # Evaluate offspring.
        for ind in offspring:
            evaluate_objectives(ind, ori_src_bytes, ori_pdg, ori_data, w2v_model, true_label, wrapper, lang, sampled_vars)
            if ind.true_conf < best_true_conf:
                best_true_conf = ind.true_conf
                best_variant = ind.adv_code
            if ind.is_success:
                if first_success_gen is None:
                    first_success_gen = gen
                if first_success_variant is None:
                    first_success_variant = ind.adv_code
                    success_true_conf = ind.true_conf
                successful_archives.append(ind)
        
        # 3. Merge and sort.
        combined_pop = population + offspring
        fronts = fast_non_dominated_sort(combined_pop)
        
        # 4. Select the next generation.
        new_population = []
        front_idx = 0
        while len(new_population) + len(fronts[front_idx]) <= POP_SIZE:
            calculate_crowding_distance(fronts[front_idx])
            new_population.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break
        
        if len(new_population) < POP_SIZE and front_idx < len(fronts):
            calculate_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            new_population.extend(fronts[front_idx][:POP_SIZE - len(new_population)])
            
        population = new_population
        # print([ind.chromosome for ind in new_population])
        
        # Simple logging.
        min_prob = min(ind.objectives[0] for ind in population)
        print(f"Gen {gen+1} Min Prob: {min_prob:.4f}, Success Count: {len(successful_archives)}")

        try:
            del offspring, combined_pop, fronts, new_population
        except UnboundLocalError:
            pass

        if first_success_gen is not None and gen - first_success_gen >= POST_SUCCESS_GENERATIONS:
            print(
                f"Early stop at generation {gen+1}: "
                f"success found at generation {first_success_gen+1}, "
                f"continued {POST_SUCCESS_GENERATIONS} extra generations."
            )
            break

    # ================= Result selection =================
    if successful_archives:
        # MOAA strategy: select among successful samples according to preferences.
        # Example here: prioritize the lowest modification rate (Obj 3), then the lowest semantic distance (Obj 2).
        successful_archives.sort(key=lambda x: (x.objectives[2], x.objectives[1]))
        best_ind = successful_archives[0]
        print(f"Attack succeeded! Prob: {best_ind.objectives[0]:.4f}, ModRate: {best_ind.objectives[2]:.2f}")
        return _build_result(
            sample_id,
            true_label,
            original_code,
            best_ind.adv_code,
            best_variant,
            first_success_variant,
            original_pred,
            original_true_conf,
            best_ind.pred_label,
            best_ind.true_conf,
            best_true_conf,
            success_true_conf,
            wrapper,
        )
    else:
        # Failed case: return the sample with the lowest probability.
        population.sort(key=lambda x: x.objectives[0])
        best_ind = population[0]
        print(f"Attack failed. Prob: {best_ind.objectives[0]:.4f}")
        return _build_result(
            sample_id,
            true_label,
            original_code,
            best_ind.adv_code,
            best_variant,
            first_success_variant,
            original_pred,
            original_true_conf,
            best_ind.pred_label,
            best_ind.true_conf,
            best_true_conf,
            success_true_conf,
            wrapper,
        )


def run_moaa_attack(
    model_name,
    checkpoint_path,
    source_code,
    true_label,
    sample_id="unknown",
    sample_i=None,
    lang='cpp',
    max_iter=MOAA_MAX_GEN,
    input_dim=100,
    output_dim=200,
):
    effective_sample_id = sample_id if sample_id != "unknown" else (
        sample_i if sample_i is not None else "unknown"
    )
    wrapper = ModelWrapper(model_name, checkpoint_path, input_dim=input_dim, output_dim=output_dim)
    return moaa_attack(
        src=source_code,
        true_label=true_label,
        wrapper=wrapper,
        lang=lang,
        max_iter=max_iter,
        sample_id=effective_sample_id,
    )


# ================= Helper functions =================

def evaluate_objectives(ind, ori_code, ori_pdg, ori_emb, w2v_model, true_label, wrapper, lang, all_vars):
    """Compute the three objectives: [Prob, SemanticDist, ModRate]."""
    
    # 1. Generate the graph data object (GlobalStorage) for the adversarial sample.
    adv_data_obj = multi_renamed_pdg_to_embedding(ori_pdg, w2v_model, ind.chromosome, true_label).to(torch.device(DEVICE))
    
    # 2. Prediction. The model wrapper usually knows how to handle GlobalStorage objects.
    pred_label, prob = wrapper.predict_label_and_true_conf(adv_data_obj, true_label)
    is_success = pred_label != true_label
    
    # 3. Compute semantic distance (Cosine Distance).
    # Bug fix: extract the feature tensor from the object. It is usually stored in .x.
    # If your GlobalStorage stores features under another name, such as .feat or .feature, modify it here.
    if hasattr(adv_data_obj, 'x'):
        adv_tensor = adv_data_obj.x
        ori_tensor = ori_emb.x
    elif hasattr(adv_data_obj, 'feature'):
        adv_tensor = adv_data_obj.feature
        ori_tensor = ori_emb.feature
    else:
        # If the object cannot be recognized, print its attributes for debugging.
        # print(dir(adv_data_obj))
        # Assume it is .x.
        adv_tensor = adv_data_obj.x
        ori_tensor = ori_emb.x

    # Compute the mean of the tensor as the semantic vector of the code (Mean Pooling).
    # This works whether the shape is [Nodes, Dim] or [1, Nodes, Dim].
    if adv_tensor.dim() > 1:
        # Average over the node dimension to obtain a [Dim] vector.
        # Assuming dim 0 is the node count, or dim 1 is the node count; flattening then averaging is usually the most robust.
        adv_vec = torch.mean(adv_tensor.float(), dim=0) 
        ori_vec = torch.mean(ori_tensor.float(), dim=0)
        
        # If there are extra dimensions, such as a batch dimension, squeeze or average again.
        if adv_vec.dim() > 1:
            adv_vec = torch.mean(adv_vec, dim=0)
            ori_vec = torch.mean(ori_vec, dim=0)
    else:
        adv_vec = adv_tensor.float()
        ori_vec = ori_tensor.float()
        
    sim = F.cosine_similarity(ori_vec.unsqueeze(0), adv_vec.unsqueeze(0)).item()
    sem_dist = 1.0 - sim # The objective is to minimize the distance.
    
    # 4. Modification rate.
    changed = sum(1 for k, v in ind.chromosome.items() if k != v)
    mod_rate = changed / len(all_vars) if all_vars else 0
    
    ind.objectives = [prob, sem_dist, mod_rate]
    ind.is_success = is_success
    ind.pred_label = pred_label
    ind.true_conf = prob
    
    # Deferred decoding.
    ind.adv_code = apply_chromosome(ori_code, ind.chromosome, lang)
    del adv_data_obj, adv_tensor, ori_tensor, adv_vec, ori_vec

def crossover(p1_chrom, p2_chrom):
    """
    Safe crossover: ensure the generated child has no duplicate values (value collision).
    """
    if random.random() > CROSSOVER_RATE:
        return p1_chrom.copy()
    
    keys = list(p1_chrom.keys())
    if len(keys) < 2: return p1_chrom.copy()
    
    point = random.randint(1, len(keys) - 1)
    child = {}
    
    # Track which names have already been used by the child to prevent many-to-one mappings.
    used_values_in_child = set()
    
    for i, key in enumerate(keys):
        # 1. Determine the source.
        source_chrom = p1_chrom if i < point else p2_chrom
        target_val = source_chrom[key]
        
        # 2. Conflict checking.
        # If target_val was already used by a previous variable and target_val is not key itself, this may be a collision.
        # The logic here is: if target_val already exists in used_values_in_child, a previous variable has been renamed to it.
        # We must not let the current key be renamed to the same value.
        
        if target_val in used_values_in_child:
            # Conflict-handling strategy:
            # A. Try the value from the other parent.
            alt_chrom = p2_chrom if i < point else p1_chrom
            alt_val = alt_chrom[key]
            
            if alt_val not in used_values_in_child:
                final_val = alt_val
            else:
                # B. If both parents conflict, try restoring the original name.
                if key not in used_values_in_child:
                    final_val = key
                else:
                    # C. Extreme case: the original name is also occupied.
                    # Abandon this crossover and return parent p1 directly to ensure consistency.
                    return p1_chrom.copy()
        else:
            final_val = target_val
            
        child[key] = final_val
        used_values_in_child.add(final_val)
        
    return child

def mutate(chromosome, candis_dict, importance_map):
    """
    Safe mutation: prevent duplicate values from being generated.
    """
    mutated = chromosome.copy()
    
    # Count all currently used values.
    current_values = set(mutated.values())
    
    for var in mutated.keys():
        # Compute the mutation rate with importance weighting.
        imp = importance_map.get(var, 0)
        prob = MUTATION_RATE * (1.0 + imp)
        prob = min(prob, 0.8) # Upper bound.
        
        if random.random() < prob:
            if var in candis_dict and candis_dict[var]:
                candidates = candis_dict[var]
                random.shuffle(candidates) # Shuffle randomly.
                
                for cand in candidates:
                    # Skip the candidate if it is already occupied by another variable and is not the current value.
                    if cand in current_values and cand != mutated[var]:
                        continue
                    
                    # Apply mutation.
                    old_val = mutated[var]
                    if old_val in current_values:
                        current_values.remove(old_val)
                    
                    mutated[var] = cand
                    current_values.add(cand)
                    break # Stop after finding a valid candidate.
    return mutated

def tournament_selection(population, k=2):
    candis = random.sample(population, k)
    # Prefer lower rank first, then higher crowding distance.
    best = candis[0]
    for c in candis[1:]:
        if c.rank < best.rank:
            best = c
        elif c.rank == best.rank and c.crowding_distance > best.crowding_distance:
            best = c
    return best

# Standard NSGA-II sorting algorithm; no modification needed.
def fast_non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p.domination_count = 0
        p.dominated_individuals = []
        for q in population:
            if dominates(p, q):
                p.dominated_individuals.append(q)
            elif dominates(q, p):
                p.domination_count += 1
        if p.domination_count == 0:
            p.rank = 0
            fronts[0].append(p)
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p.dominated_individuals:
                q.domination_count -= 1
                if q.domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

def dominates(ind1, ind2):
    # MOAA strategy: successful attacks have priority.
    if ind1.is_success and not ind2.is_success: return True
    if not ind1.is_success and ind2.is_success: return False
    
    # Pareto dominance (minimization).
    better_any = False
    for o1, o2 in zip(ind1.objectives, ind2.objectives):
        if o1 > o2: return False
        if o1 < o2: better_any = True
    return better_any

def calculate_crowding_distance(front):
    l = len(front)
    if l == 0: return
    for ind in front: ind.crowding_distance = 0
    num_objs = len(front[0].objectives)
    for m in range(num_objs):
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')
        r = front[-1].objectives[m] - front[0].objectives[m]
        if r == 0: continue
        for i in range(1, l-1):
            front[i].crowding_distance += (front[i+1].objectives[m] - front[i-1].objectives[m]) / r
    
                
def apply_chromosome(code, chromosome, lang='cpp'):
    """
    Apply the replacement table to the code.
    chromosome: dict, e.g., {'sushu': 'cnt', 'res': 'ans'}
    """
    # Replace all variables by position in one pass.
    temp_code = rename_identifiers(code, chromosome, lang)
    return temp_code

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
    result = run_moaa_attack(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        source_code=ori_code,
        true_label=0,
        sample_id="demo_moaa",
        lang='cpp',
        max_iter=10,
        input_dim=100,
        output_dim=200,
    )
    print(result.to_dict())


if __name__ == "__main__":
    test()