import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

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
    pdg2embedding,
    renamed_pdg_to_embedding,
    src2pdg,
)
from common.utils.renamer import rename_identifier
from src.attack.masking import importance_sampling
from src.model.wrapper import ModelWrapper

S = get_settings()
DEVICE = S.device
MHM_MAX_ITER = S.mhm_max_iter
MODEL_NAME = "reveal"
CHECKPOINT_PATH = S.reveal_checkpoint

# ================= MHM resource-control parameters =================
# Does not affect upstream function calls; defaults are used when settings fields are absent.

# Maximum number of queries per sample. <=0 means unlimited.
MHM_MAX_QUERIES = int(getattr(S, "mhm_max_queries", 500))

# Verbose logging switch. Use False for full-scale experiments.
MHM_VERBOSE = bool(getattr(S, "mhm_verbose", False))

# Sample-level progress logging switch.
MHM_PROGRESS = bool(getattr(S, "mhm_progress", True))


def _vprint(*args, **kwargs):
    """Candidate-level and iteration-level verbose logs; disabled by default for formal runs."""
    if MHM_VERBOSE:
        print(*args, **kwargs)


def _pprint(*args, **kwargs):
    """Sample-level key logs."""
    if MHM_PROGRESS:
        print(*args, **kwargs)


def _budget_exceeded(wrapper: ModelWrapper) -> bool:
    return MHM_MAX_QUERIES > 0 and wrapper.get_query_count() >= MHM_MAX_QUERIES

@lru_cache(maxsize=8)
def load_global_vocab(vocab_path):
    """
    Load the global vocab and cache it by path.

    Return a tuple instead of a set:
    - Avoid repeatedly reading JSON for each attacked sample.
    - Avoid running list(vocab - set(vars)) in every MHM iteration.
    - random.choice(tuple) supports direct O(1) sampling.
    """
    import json

    vocab_path = str(Path(vocab_path).expanduser().resolve())

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)

    global_vocab = set()
    for item in vocab_list:
        if isinstance(item, dict):
            global_vocab.update(item.keys())
        else:
            global_vocab.add(item)

    # Sorting is not required, but it stabilizes tuple order for reproducibility.
    return tuple(global_vocab)


def _ensure_vocab_tuple(vocab):
    """
    Keep compatibility when external callers still pass a set/list/tuple.
    No upstream call changes are required.
    """
    if vocab is None:
        return tuple()
    if isinstance(vocab, tuple):
        return vocab
    return tuple(vocab)


def _sample_new_identifier(vocab_tuple, forbidden_names, max_trials=64):
    """
    Randomly sample a new identifier from vocab_tuple that does not conflict with current variables.

    Compared with the original random.choice(list(vocab - set(vars))):
    - Avoid constructing a large list in every iteration.
    - Use only a small number of random retries.
    - Fall back to a linear scan only in extreme cases.
    """
    if not vocab_tuple:
        return None

    for _ in range(max_trials):
        cand = random.choice(vocab_tuple)
        if cand not in forbidden_names:
            return cand

    # Fallback: if repeated samples hit forbidden names, find one by linear scan.
    for cand in vocab_tuple:
        if cand not in forbidden_names:
            return cand

    return None

def _build_result(
    sample_id,
    model_name,
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
        attack_name="mhm",
        model_name=model_name,
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


def mhm_attack(
    src,
    true_label,
    wrapper,
    vocab,
    lang='cpp',
    max_iter=MHM_MAX_ITER,
    sample_id="unknown",
    model_name=MODEL_NAME,
):
    if isinstance(src, str):
        current_code = src.encode('utf-8')
    else:
        current_code = src

    wrapper.reset_query_count()
    original_code = to_text(current_code)
    renamed_vars = []

    # Accept set/list/tuple vocab inputs and convert internally to a tuple to avoid large list construction per iteration.
    vocab_tuple = _ensure_vocab_tuple(vocab)

    w2v_model = load_word_vectors()
    current_pdg = src2pdg(current_code)
    current_data = pdg2embedding(
        current_pdg, w2v_model, true_label
    ).to(torch.device(DEVICE))

    original_pred, original_true_conf = wrapper.predict_label_and_true_conf(
        current_data, true_label
    )

    # Maintain the current state's true_conf to compute alpha directly later and avoid repeated forwards in compute_acceptance.
    current_true_conf = original_true_conf

    best_variant = current_code
    best_true_conf = original_true_conf
    first_success_variant = None
    success_true_conf = None

    if original_pred != true_label:
        return _build_result(
            sample_id,
            model_name,
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

    # MHM Iter
    for k in range(max_iter):
        if _budget_exceeded(wrapper):
            _pprint(f"[Budget] Stop MHM: query_count={wrapper.get_query_count()}")
            break

        # extract identifier & importance sampling
        var_probs = importance_sampling(current_code, true_label, wrapper)

        # Keep at most the top 20 variables; keep all variables if fewer than 20 are available.
        # Preserve the original logic without further changing the search space.
        var_probs = var_probs[:20]

        if not var_probs:
            break

        # Separate variable names and probabilities.
        vars = np.array([item[0] for item in var_probs])
        probs = np.array([item[1] for item in var_probs])

        # Ensure variables that have already been renamed are not renamed again.
        valid_indices = ~np.isin(vars, renamed_vars)
        valid_vars = vars[valid_indices]
        valid_probs = probs[valid_indices]

        if len(valid_vars) == 0:
            break

        probs_sum = valid_probs.sum()
        if probs_sum <= 0:
            # Fall back to uniform sampling in extreme cases to avoid division by zero.
            valid_probs = np.ones_like(valid_probs, dtype=float) / len(valid_probs)
        elif probs_sum != 1:
            valid_probs = valid_probs / probs_sum

        target_id = np.random.choice(valid_vars, p=valid_probs)
        renamed_vars.append(target_id)

        # The replacement identifier must not conflict with current important variables.
        # The original logic was random.choice(list(vocab - set(vars))).
        # Now use random retries over the tuple to avoid constructing a large list in every iteration.
        forbidden_names = set(vars.tolist())
        new_id = _sample_new_identifier(vocab_tuple, forbidden_names)

        if new_id is None:
            _vprint(f"Iter {k + 1}: no valid new identifier for '{target_id}'")
            continue

        _vprint(f"Iter {k + 1}: Trying to rename '{target_id}' to '{new_id}'")

        # Rename the identifier in the source code
        proposed_code = rename_identifier(current_code, target_id, new_id, lang).encode('utf-8')

        proposed_data = renamed_pdg_to_embedding(
            current_pdg, w2v_model, target_id, new_id, true_label
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
            # Release the proposed_data reference before returning after a successful attack.
            del proposed_data

            return _build_result(
                sample_id,
                model_name,
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

        # This previously called wrapper.compute_acceptance(true_label, current_data, proposed_data).
        # That would run one additional forward pass for each of current_data and proposed_data.
        # Now compute alpha directly from the existing current_true_conf / proposed_true_conf.
        if proposed_true_conf <= 0:
            alpha = 1.0
        else:
            alpha = min(1.0, current_true_conf / proposed_true_conf)

        _vprint(f"Acceptance ratio (alpha): {alpha}")

        u = random.uniform(0, 1)
        _vprint(f"Random value (u): {u}")

        if u < alpha:
            # Accept proposed: release the old current_data and keep proposed_data as the new current_data.
            old_current_data = current_data

            current_code = proposed_code
            current_data = proposed_data
            current_true_conf = proposed_true_conf

            del old_current_data
        else:
            # Reject proposed: immediately release proposed_data to avoid residual GPU Data.
            del proposed_data
        # identifiers = extract_identifiers_from_one_src(current_code, lang) # test
    
    final_pred, final_true_conf = wrapper.predict_label_and_true_conf(current_data, true_label)
    if final_true_conf < best_true_conf:
        best_true_conf = final_true_conf
        best_variant = current_code
    # Release the current GPU Data reference after the final prediction.
    del current_data
    return _build_result(
        sample_id,
        model_name,
        true_label,
        original_code,
        current_code,
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


def run_mhm_attack(
    model_name: str,
    checkpoint_path: str,
    source_code: Union[str, bytes],
    true_label: int,
    sample_id: Union[str, int] = "unknown",
    sample_i: Optional[Union[str, int]] = None,
    vocab: Optional[set] = None,
    vocab_path: Optional[str] = None,
    lang: str = "cpp",
    max_iter: int = MHM_MAX_ITER,
    input_dim: int = 100,
    output_dim: int = 200,
) -> AttackResult:
    """
    Unified external interface for experiment scripts (MHM only).

    Required core parameters:
    - model_name / checkpoint_path
    - source_code / true_label
    - sample_id (compatible with sample_i)

    MHM-specific parameters (vocab, max_iter, lang) keep their default values.
    """
    effective_sample_id = sample_id if sample_id != "unknown" else (
        sample_i if sample_i is not None else "unknown"
    )
    if vocab is None:
        resolved_vocab_path = vocab_path or str(Path(S.vocab_dir) / "ori_src_vocab.json")
        vocab = load_global_vocab(resolved_vocab_path)

    wrapper = ModelWrapper(model_name, checkpoint_path, input_dim=input_dim, output_dim=output_dim)
    return mhm_attack(
        src=source_code,
        true_label=true_label,
        wrapper=wrapper,
        vocab=vocab,
        lang=lang,
        max_iter=max_iter,
        sample_id=effective_sample_id,
        model_name=model_name,
    )

def test():
    vocab = load_global_vocab(
        str(Path(S.vocab_dir) / "ori_src_vocab.json")
    )
    # print(f"Loaded vocab size: {len(vocab)}")
    # print(random.sample(vocab, 10))
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
    result = run_mhm_attack(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        source_code=normal_code,
        true_label=1,
        sample_id="demo_mhm",
        vocab=vocab,
        lang='cpp',
        max_iter=10,
        input_dim=100,
        output_dim=200,
    )
    print(result.to_dict())


if __name__ == "__main__":
    test()