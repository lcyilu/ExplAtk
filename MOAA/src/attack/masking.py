import sys
from pathlib import Path

_METHOD_ROOT = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_METHOD_ROOT, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
from common.config_loader import get_settings
from common.utils.gen_embedding import src2embedding, src2pdg, pdg2embedding, load_word_vectors, renamed_pdg_to_embedding
from common.utils.parser import extract_identifiers_from_one_src
from common.utils.renamer import rename_identifier
from src.model.wrapper import ModelWrapper

S = get_settings()
DEVICE = S.device
MASK_PLACEHOLDER = S.mask_placeholder
MASK = S.mask


def importance_sampling(src, true_label, wrapper: ModelWrapper, lang='cpp'):
    if isinstance(src, str):
        ori_code = src.encode('utf-8')
    else:
        ori_code = src
    # importance sampling for the identifier in the source code
    raw_identifiers = extract_identifiers_from_one_src(ori_code, lang)
    unique_identifiers = list(set(raw_identifiers)) 

    w2v_model = load_word_vectors()

    ori_pdg = src2pdg(ori_code)
    ori_data = pdg2embedding(ori_pdg, w2v_model, true_label).to(torch.device(DEVICE))
    # ori_data = src2embedding(ori_code, true_label).to(torch.device(DEVICE))
    ori_prob = wrapper.predict_prob(ori_data, true_label)

    # Store results as {identifier: importance_score}.
    identifier_scores = {}

    print("-------- Identifiters Importance Samplig -----------")
    for index,id_name in enumerate(unique_identifiers):
        mask_id = MASK_PLACEHOLDER

        # proposed_code = rename_identifier(ori_code, id_name, mask_id, lang).encode('utf-8')
        # proposed_data = src2embedding(proposed_code, true_label).to(torch.device(DEVICE))
        proposed_data = renamed_pdg_to_embedding(ori_pdg, w2v_model, id_name, MASK, true_label).to(torch.device(DEVICE))

        score = wrapper.compute_importance(true_label, ori_prob, proposed_data)
        print(f"Enum {index+1}: Trying to rename '{id_name}' to '{mask_id}', id score: {score}") # test
        identifier_scores[id_name] = score
    
    # ALERT greedy strategy: directly return the list sorted by score.
    sorted_list = sorted(identifier_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_list
    
    # MHM MCMC sampling: Softmax probabilities need to be computed.
    # ids = list(identifier_scores.keys())
    # scores = np.array(list(identifier_scores.values()))
    
    # # Edge-case check: if the sum of all scores is 0 or very close to 0.
    # if np.sum(scores) < 1e-9:
    #     # If all variables are unimportant, assign them equal probability with a uniform distribution.
    #     probs = np.ones(len(scores)) / len(scores)
    # else:
    #     T = 0.1  # Amplify differences so important variables are more likely to be selected.
    #     scaled_scores = scores / T
        
    #     # Subtract the maximum to prevent numerical overflow, which is a good practice even if scores are non-negative and usually small.
    #     exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    #     probs = exp_scores / np.sum(exp_scores)
    
    # # Combine identifiers and probabilities.
    # importance_probs = list(zip(ids, probs))
    
    # # Sort by probability from high to low for easier inspection, even though sampling is random.
    # importance_probs.sort(key=lambda x: x[1], reverse=True)

    # return importance_probs