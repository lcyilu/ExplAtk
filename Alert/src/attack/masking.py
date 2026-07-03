
import numpy as np
import torch
from src.utils.gen_embedding import src2embedding, src2pdg, pdg2embedding, load_word_vectors, renamed_pdg_to_embedding
from src.utils.parser import extract_identifiers_from_one_src
from src.utils.renamer import rename_identifier
from src.config import DEVICE, MASK_PLACEHOLDER, MASK
from src.model.wrapper import ModelWrapper

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

    # Store results as {identifier: importance_score}
    identifier_scores = {}

    print("-------- Identifier Importance Sampling -----------")
    for index,id_name in enumerate(unique_identifiers):
        mask_id = MASK_PLACEHOLDER

        # proposed_code = rename_identifier(ori_code, id_name, mask_id, lang).encode('utf-8')
        # proposed_data = src2embedding(proposed_code, true_label).to(torch.device(DEVICE))
        proposed_data = renamed_pdg_to_embedding(ori_pdg, w2v_model, id_name, MASK, true_label).to(torch.device(DEVICE))

        score = wrapper.compute_importance(true_label, ori_prob, proposed_data)
        print(f"Enum {index+1}: Trying to rename '{id_name}' to '{mask_id}', id score: {score}") # test
        identifier_scores[id_name] = score
    
    # ALERT (greedy strategy): directly return the list sorted by score
    sorted_list = sorted(identifier_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_list
    
    # MHM (MCMC sampling): compute Softmax probabilities
    # ids = list(identifier_scores.keys())
    # scores = np.array(list(identifier_scores.values()))
    
    # # Edge-case check: if the sum of all scores is 0 (or very close to 0)
    # if np.sum(scores) < 1e-9:
    #     # If none of the variables is important, assign equal probabilities (uniform distribution)
    #     probs = np.ones(len(scores)) / len(scores)
    # else:
    #     T = 0.1  # Amplify score differences so important variables are more likely to be selected
    #     scaled_scores = scores / T
        
    #     # Subtract the maximum to prevent numerical overflow (usually unnecessary when score >= 0 and small, but this is good practice)
    #     exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
    #     probs = exp_scores / np.sum(exp_scores)
    
    # # Pair identifiers with their probabilities
    # importance_probs = list(zip(ids, probs))
    
    # # Sort by probability in descending order for easier inspection (sampling remains random, but sorting improves readability)
    # importance_probs.sort(key=lambda x: x[1], reverse=True)

    # return importance_probs