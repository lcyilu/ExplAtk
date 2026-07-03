import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path

_METHOD_ROOT = Path(__file__).resolve().parents[2]
_ROOT = Path(__file__).resolve().parents[3]
for _p in (_METHOD_ROOT, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn.functional as F
import numpy as np
import random
import re
import json
from transformers import AutoTokenizer, AutoModel
from gensim.models import FastText
from common.attack_result import AttackResult, to_text
from common.config_loader import get_settings
from common.utils.parser import initialize_language_parser, extract_identifiers_from_one_src, is_not_keyword
from common.utils.gen_embedding import src2embedding, src2pdg, pdg2embedding, load_word_vectors, renamed_pdg_to_embedding, multi_renamed_pdg_to_embedding
from common.utils.renamer import rename_identifier, rename_identifiers
from src.model.wrapper import ModelWrapper
from common.ast_parser.run_parser import change_code_style, get_code_style
import threading

# ── Attacker instance cache; key = initialization parameter tuple; reused at victim level ──
_ATTACKER_CACHE: dict = {}
_ATTACKER_CACHE_LOCK = threading.Lock()


def _get_or_create_attacker(
    codebert_path: str,
    code_db_path: str,
    src_paths_path: str,
    fasttext_path: str,
    device: str,
    lang: str,
) -> "CODAAttacker":
    """
    Look up the cache by initialization parameters. Reuse the cached instance on a hit;
    otherwise create it, store it in the cache, and ensure it is initialized only once across threads.
    """
    cache_key = (codebert_path, code_db_path, src_paths_path, fasttext_path, device, lang)

    # First check without a lock for the fast path.
    if cache_key in _ATTACKER_CACHE:
        return _ATTACKER_CACHE[cache_key]

    # Second check with a lock to prevent duplicate creation under concurrency.
    with _ATTACKER_CACHE_LOCK:
        if cache_key not in _ATTACKER_CACHE:
            print(f"[CODA] Creating new CODAAttacker for key: "
                  f"({code_db_path}, lang={lang})")
            _ATTACKER_CACHE[cache_key] = CODAAttacker(
                codebert_path=codebert_path,
                code_db_path=code_db_path,
                src_paths_path=src_paths_path,
                fasttext_path=fasttext_path,
                device=device,
                lang=lang,
            )
        return _ATTACKER_CACHE[cache_key]

S = get_settings()
LOCAL_CODEBERT_PATH = S.local_codebert_path
CODA_DB_DIR = Path(S.coda_db_dir)
CODA_CODE_DB = S.coda_code_db
CODA_SRC_PATHS = S.coda_src_paths
MODEL_NAME = "ivdetect"
CHECKPOINT_PATH = S.ivdetect_checkpoint
WV = load_word_vectors()

def get_coda_db_paths(model_name: str, dataset_name: str):
    base = CODA_DB_DIR / f"{model_name.lower()}_{dataset_name.lower()}"
    return str(base) + "_cls.pt", str(base) + "_src_paths.json"

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
        attack_name="coda",
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

class CODAAttacker:
    def __init__(self, 
                 codebert_path=LOCAL_CODEBERT_PATH, 
                 code_db_path=CODA_CODE_DB, 
                 src_paths_path=CODA_SRC_PATHS,
                 fasttext_path='{HOME_PATH}/vul_robustness/CODA/data/saved_models/fasttext/ori_universal_fasttext.model', 
                 device='cuda',
                 lang='cpp'):
        """
        Main CODA attacker class: integrates vulnerable-location awareness, perturbation material selection, adversarial sample generation, and injection.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lang = lang
        
        # 1. Load the proxy model (CodeBERT) for vector and attention computation.
        print(f"[CODA] Loading Proxy Model (CodeBERT) from {codebert_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_path)
        self.proxy_model = AutoModel.from_pretrained(codebert_path).to(self.device)
        self.proxy_model.eval()
        
        # 2. Initialize the Tree-sitter parser.
        print(f"[CODA] Initializing Tree-sitter Parser for {lang}...")
        self.parser = initialize_language_parser(lang)
        
        # 3. Load the code database (Dissimilar Code DB).
        print(f"[CODA] Loading Dissimilar Code Database...")
        if os.path.exists(code_db_path) and os.path.exists(src_paths_path):
            # self.db_vecs = torch.load(code_db_path, map_location=self.device)
            self.db_vecs = torch.load(code_db_path, map_location="cpu")
            # Pre-normalize vectors to speed up later computations.
            self.db_vecs = F.normalize(self.db_vecs, p=2, dim=1)
            
            with open(src_paths_path, 'r', encoding='utf-8') as f:
                self.db_paths = json.load(f)
            print(f"[CODA] Database loaded. Size: {len(self.db_paths)}")
        else:
            print(f"[Error] DB files not found at {code_db_path} or {src_paths_path}")
            self.db_vecs = None
            self.db_paths = []

        self.fasttext_model = None
        if os.path.exists(fasttext_path):
            print(f"[CODA] Loading FastText model from {fasttext_path}...")
            try:
                # Load the full FastText model, including subword information.
                self.fasttext_model = FastText.load(fasttext_path)
                print(f"[CODA] FastText model loaded. Vocab size: {len(self.fasttext_model.wv)}")

                # ── Added: preload the word-vector matrix onto the GPU ──────────────────────────────
                vocab        = self.fasttext_model.wv.index_to_key          # List[str], vocabulary ordered by index
                self.ft_vocab       = vocab
                self.ft_word2idx    = {w: i for i, w in enumerate(vocab)}   # word -> row index

                # Convert the numpy array [vocab_size, dim] to a GPU Tensor and L2-normalize it.
                vectors_np   = self.fasttext_model.wv.vectors               # shape: [V, D]
                vectors_t    = torch.tensor(vectors_np, dtype=torch.float32, device=self.device)
                # L2-normalize so later dot products directly produce cosine similarity.
                self.ft_vectors = F.normalize(vectors_t, p=2, dim=1)        # [V, D], kept on the GPU
                print(f"[CODA] FastText vectors moved to {self.device}. Shape: {self.ft_vectors.shape}")

            except Exception as e:
                print(f"[Error] Failed to load FastText model: {e}")
        else:
            print(f"[Warning] FastText model file not found at: {fasttext_path}")
            print("[Warning] Similarity calculation will return 0.0")


    def _get_code_vector(self, code_str):
        """Compute the CodeBERT [CLS] vector."""
        inputs = self.tokenizer(code_str, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.proxy_model(**inputs)
            return outputs.last_hidden_state[:, 0, :] # [1, 768]
        
    # ================= Select Top-N similar code snippets (Similar Code) =================
    def _select_reference_inputs(self, target_code, N=10):
        """
        CODA phase 1: reference input selection.
        
        Args:
            target_code (str): Source code string to attack. Ideally, identifiers should be <mask>-processed to match the CODA design.
            N (int): Number of reference code samples to select. The paper uses 10 by default.
            
        Returns:
            List[str]: Paths to the N most similar code files.
        """
        if self.db_vecs is None or len(self.db_paths) == 0:
            print("[Error] Database is empty or not loaded.")
            return []

        # 1. Get the vector for the target code.
        # target_vec = self._get_code_vector(target_code) # shape: [1, hidden_dim]
        target_vec = self._get_code_vector(target_code).cpu()  # Move to CPU.

        # 2. Compute similarity (Cosine Similarity).
        # Since both target_vec and self.db_vecs are normalized, A . B^T is equivalent to cosine similarity.
        # self.db_vecs shape: [DB_Size, hidden_dim]
        # scores shape: [1, DB_Size]
        scores = torch.mm(target_vec, self.db_vecs.T).squeeze(0)

        # 3. Select the Top-N most similar entries.
        # largest=True selects the largest values (most similar), and sorted=True sorts scores from high to low.
        # Ensure that N does not exceed the total database size.
        k = min(N, len(self.db_paths))
        top_scores, top_indices = torch.topk(scores, k=k, largest=True, sorted=True)

        # 4. Retrieve the corresponding file paths.
        selected_paths = []
        for idx in top_indices:
            file_idx = idx.item() # Convert to a Python int.
            selected_paths.append(self.db_paths[file_idx])
            
        # Optional: print the highest similarity score for debugging.
        # print(f"[CODA] Top similarity score: {top_scores[0].item():.4f}")

        return selected_paths

    def _est_transform(self, target_code, reference_paths):
        """
        CODA phase 2: equivalent structure transformation (EST).
        Modify the target code structure based on the style distribution of the reference code.
        
        Args:
            target_code (str): Original target code.
            reference_paths (list): File paths returned by select_reference_inputs.
            
        Returns:
            list: A list containing multiple structurally transformed code versions.
        """
        # 1. Read the contents of the reference code files.
        reference_contents = []
        for path in reference_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    reference_contents.append(f.read())
            except Exception as e:
                print(f"[Warn] Failed to read {path}: {e}")
                continue
        
        if not reference_contents:
            print("[Warn] No reference content available. Skipping EST.")
            return [target_code]

        # 2. Compute the style distribution of the reference code by calling get_code_style.
        # code_style is a probability list, for example [[0.2, 0.8], ...].
        target_style = get_code_style(reference_contents, 'c' if self.lang=='cpp' else self.lang)
        
        # 3. Prepare the candidate variable-name pool.
        # change_code_style needs this list to create new variables, such as when extracting constants into variables.
        # This generates a batch of generic temporary variable names; they can also be extracted from reference_contents.
        candidate_vars = [f"temp_var_{i}" for i in range(100)]
        
        # 4. Run the structure transformation by calling change_code_style.
        # Note: the provided function uses random internally, so consider calling it multiple times or accepting all returned variants.
        # change_code_style returns a list: [original, transformed_v1, transformed_v2, ...].
        transformed_codes = change_code_style(
            source_code=target_code, 
            lang='c' if self.lang=='cpp' else self.lang, 
            all_variable_name=candidate_vars, 
            code_style=target_style
        )
        
        # Deduplicate and keep transformations other than the original code.
        unique_codes = list(set(transformed_codes[-3:])) # Take the last three variants, assuming they have different structures.
        
        print(f"[CODA] EST phase generated {len(unique_codes)} variants.")
        return unique_codes

    def _calculate_fasttext_similarity(self, word_a, word_b):
        """
        Compute the FastText semantic similarity between two identifiers.
        """
        # 1. Check whether the model has been loaded.
        if self.fasttext_model is None:
            return 0.0
        
        # 2. Basic filtering: return 0 directly if either word is empty.
        if not word_a or not word_b:
            return 0.0

        # 3. Compute similarity.
        try:
            # self.fasttext_model.wv is a KeyedVectors object.
            # The .similarity() method automatically handles OOV (Out-of-Vocabulary) cases.
            # As long as word_a/word_b can be decomposed into n-grams seen during training, such as 'count' -> 'cou', 'oun',
            # it can compute a non-zero similarity score.
            score = self.fasttext_model.wv.similarity(word_a, word_b)
            return float(score)
            
        except KeyError:
            # Rare case: if a word is too short and contains completely unseen characters, such as rare symbols,
            # no n-grams can be generated, and Gensim may raise a KeyError.
            return 0.0
        
    def _rank_substitutions(self, tgt_list, cand_list):
        """
        Perform vector lookup and cosine-similarity matrix computation entirely on the GPU,
        and return [(score, tgt_var, cand_var), ...] sorted by descending score.
        """
        # ── Fall back to the original logic when no preloaded matrix is available ────────────────────────────
        if not hasattr(self, 'ft_vectors') or self.ft_vectors is None:
            return [(0.0, t, cand_list[0]) for t in tgt_list if cand_list]

        # ── 1. Use word2idx to look up row indices and filter OOV entries ─────────────────────
        def _lookup(words):
            valid, idxs = [], []
            for w in words:
                idx = self.ft_word2idx.get(w)   # OOV → None
                if idx is not None:
                    valid.append(w)
                    idxs.append(idx)
            return valid, idxs

        valid_tgts,  tgt_idxs  = _lookup(tgt_list)
        valid_cands, cand_idxs = _lookup(cand_list)

        if not tgt_idxs or not cand_idxs:
            return []

        # ── 2. Slice submatrices from the preloaded matrix by index; zero-copy and directly on GPU ──
        tgt_idx_t  = torch.tensor(tgt_idxs,  dtype=torch.long, device=self.device)
        cand_idx_t = torch.tensor(cand_idxs, dtype=torch.long, device=self.device)

        tgt_vecs  = self.ft_vectors[tgt_idx_t]   # [|tgt|,  D]
        cand_vecs = self.ft_vectors[cand_idx_t]  # [|cand|, D]
        # ft_vectors has already been L2-normalized, so dot product == cosine similarity.

        # ── 3. Run one matrix multiplication to obtain the full similarity matrix ────────────────────
        sim_matrix = torch.mm(tgt_vecs, cand_vecs.T)  # [|tgt|, |cand|], on GPU

        # ── 4. Select the best candidate for each target word ──────────────────────────────
        best_scores, best_j = sim_matrix.max(dim=1)   # Maximum value of each row and its column index.

        # Move results back to CPU only once; only scalar results are moved, not the matrix.
        best_scores_cpu = best_scores.cpu().tolist()
        best_j_cpu      = best_j.cpu().tolist()

        substitution_pairs = [
            (best_scores_cpu[i], valid_tgts[i], valid_cands[best_j_cpu[i]])
            for i in range(len(valid_tgts))
        ]

        # ── 5. Sort by descending score ─────────────────────────────────────
        substitution_pairs.sort(key=lambda x: x[0], reverse=True)
        return substitution_pairs


    def _irt_phase(self, target_code, reference_paths, wrapper:ModelWrapper, true_label, best_state):
        """
        CODA phase 3: identifier renaming transformation (IRT).
        
        Args:
            target_code (str): Code from the EST phase, i.e. a variant.
            reference_paths (list): File path list for the reference code.
            wrapper (object): Attack wrapper containing attack_success and predict_label.
            
        Returns:
            tuple[bool, str, int, float]: (success flag, current accumulated code, predicted label, true_label confidence).
        """
        print("[CODA] Starting IRT phase...")
        
        # 0. Preprocess: ensure target_code is a string and convert it to bytes for extraction.
        if isinstance(target_code, bytes):
            target_code_str = target_code.decode('utf-8', errors='ignore')
            target_code_bytes = target_code
        else:
            target_code_str = target_code
            target_code_bytes = target_code.encode('utf-8', errors='ignore')

        # 1. Extract target identifiers (Target Identifiers).
        # Call the provided function; note that it requires bytes.
        # extract_identifiers_from_one_src returns a list, so convert it to a set for deduplication.
        tgt_ids = set(extract_identifiers_from_one_src(target_code_bytes, lang=self.lang))
        
        # 3. Extract and filter candidate identifiers (Candidate Identifiers).
        ref_ids = set()
        for path in reference_paths:
            try:
                with open(path, 'rb') as f: # Read as bytes.
                    content = f.read()
                    # Extract variable names from the reference code.
                    ids = extract_identifiers_from_one_src(content, lang=self.lang)
                    ref_ids.update(ids)
            except Exception as e:
                continue
        
        # Core logic: keep only variable names that appear in the reference code but not in the target code, introducing new features.
        candidate_ids = list(ref_ids - tgt_ids)
        
        if not candidate_ids or not tgt_ids:
            print("[CODA] No valid identifiers for renaming.")
            final_data = src2embedding(target_code_str, true_label)
            final_pred, final_true_conf = wrapper.predict_label_and_true_conf(final_data, true_label)
            if final_true_conf < best_state["best_true_conf"]:
                best_state["best_true_conf"] = final_true_conf
                best_state["best_variant"] = target_code_str
            return False, target_code_str, final_pred, final_true_conf

        # 4. Compute and sort similarities (Similarity Ranking).
        # Generate the substitution-pair list: [(score, target_var, candidate_var), ...].
        # substitution_pairs = []
        
        # print(f"[CODA] Ranking {len(tgt_ids)} target vars against {len(candidate_ids)} candidates...")
        # for t_var in tgt_ids:
        #     for c_var in candidate_ids:
        #         # Compute similarity.
        #         score = self._calculate_fasttext_similarity(t_var, c_var)
        #         substitution_pairs.append((score, t_var, c_var))
        
        # # Sort by similarity from high to low and prefer replacing the most natural variables first.
        # substitution_pairs.sort(key=lambda x: x[0], reverse=True)
        substitution_pairs = self._rank_substitutions(list(tgt_ids), candidate_ids)
        
        # 5. Iterative attack.
        current_code = target_code_str
        current_pdg = src2pdg(current_code)
        # Record target variables that have already been replaced to avoid replacing the same variable again.
        replaced_targets = set()
        
        # Limit the maximum number of attempts to avoid infinite loops or excessive runtime; adjust as needed.
        MAX_ATTEMPTS = 50 
        attempts = 0
        
        renamed_vars = {}
        for score, t_var, c_var in substitution_pairs:
            if attempts >= MAX_ATTEMPTS:
                break
            
            # Skip this target variable if it has already been replaced, because t_var no longer exists in the code.
            if t_var in replaced_targets:
                continue
                
            # Perform replacement.
            # Call the provided rename_identifier function.
            renamed_vars[t_var]=c_var
            temp_code = rename_identifier(current_code,t_var,c_var)
            
            # 6. Check whether the attack succeeds.
            # Convert the new code to model input.
            proposed_data = multi_renamed_pdg_to_embedding(current_pdg,WV,renamed_vars,1)
            pred_label, true_conf = wrapper.predict_label_and_true_conf(proposed_data, true_label)
            is_success = pred_label != true_label
            if true_conf < best_state["best_true_conf"]:
                best_state["best_true_conf"] = true_conf
                best_state["best_variant"] = temp_code
            if best_state["first_success_variant"] is None and is_success:
                best_state["first_success_variant"] = temp_code
                best_state["success_true_conf"] = true_conf
            
            if is_success:
                print(f"[CODA] Attack Success! Replaced '{t_var}' with '{c_var}' (Sim: {score:.4f})")
                return True, temp_code, pred_label, true_conf
            
            # Accumulation strategy: even if the attack does not succeed, keep this change and stack the next replacement.
            # This gradually pushes the code toward the decision boundary.
            current_code = temp_code
            replaced_targets.add(t_var)
            attempts += 1
            
        print("[CODA] IRT phase finished. Attack did not fully succeed or hit limit.")
        final_data = src2embedding(current_code, true_label)
        final_pred, final_true_conf = wrapper.predict_label_and_true_conf(final_data, true_label)
        if final_true_conf < best_state["best_true_conf"]:
            best_state["best_true_conf"] = final_true_conf
            best_state["best_variant"] = current_code
        return False, current_code, final_pred, final_true_conf
   

    # ================= Main attack interface =================

    def attack(self, target_code, wrapper:ModelWrapper, target_label, sample_id="unknown"):
        """
        Run the full CODA attack pipeline.
        
        Args:
            target_code (str): Source code string to attack.
            wrapper (ModelWrapper): Model wrapper containing predict_label and attack_success.
            target_label (int): Ground-truth label of the target code.
            max_positions (int): Optional maximum number of variables to process, used to limit computation.
            
        Returns:
            AttackResult: Unified attack result object.
        """
        print("\n" + "="*40)
        print(f"[CODA] Start Attack on Label {target_label}")
        
        wrapper.reset_query_count()
        # --- Step 0: Sanity check ---
        # Ensure the input is a string.
        if isinstance(target_code, bytes):
            target_code = target_code.decode('utf-8', errors='ignore')
            
        original_code = target_code
        initial_data = src2embedding(target_code, target_label)
        original_pred, original_true_conf = wrapper.predict_label_and_true_conf(initial_data, target_label)
        best_state = {
            "best_variant": target_code,
            "best_true_conf": original_true_conf,
            "first_success_variant": None,
            "success_true_conf": None,
        }
        if original_pred != target_label:
            print("[CODA] Skip: Model already predicts incorrectly.")
            return _build_result(
                sample_id,
                target_label,
                original_code,
                target_code,
                best_state["best_variant"],
                best_state["first_success_variant"],
                original_pred,
                original_true_conf,
                original_pred,
                original_true_conf,
                best_state["best_true_conf"],
                best_state["success_true_conf"],
                wrapper,
            )

        # --- Step 1: Reference input selection ---
        print("[CODA] Phase 1: Selecting Reference Inputs...")
        # Call the selection function defined above to get Top-N similar code paths.
        # Assume N=10 has already been set inside select_reference_inputs.
        reference_paths = self._select_reference_inputs(target_code, N=10)
        
        if not reference_paths:
            print("[Warning] No reference inputs found. Skipping EST/IRT.")
            return _build_result(
                sample_id,
                target_label,
                original_code,
                target_code,
                best_state["best_variant"],
                best_state["first_success_variant"],
                original_pred,
                original_true_conf,
                original_pred,
                original_true_conf,
                best_state["best_true_conf"],
                best_state["success_true_conf"],
                wrapper,
            )
        print(f"[CODA] Selected {len(reference_paths)} reference files.")

        # --- Step 2: Equivalent structure transformation ---
        print("[CODA] Phase 2: EST (Structure Transformation)...")
        # Get the list of structural variants.
        est_variants = self._est_transform(target_code, reference_paths[:5])
        print(f"[CODA] Generated {len(est_variants)} structural variants.")
        
        final_variant = target_code
        
        # Filter variants: check each structural variant first; if EST succeeds directly, IRT is not needed.
        candidates_for_irt = []
        
        for idx, variant in enumerate(est_variants):
            # Check whether the variant attack succeeds.
            var_data = src2embedding(variant, target_label)
            pred_label, var_true_conf = wrapper.predict_label_and_true_conf(var_data, target_label)
            if var_true_conf < best_state["best_true_conf"]:
                best_state["best_true_conf"] = var_true_conf
                best_state["best_variant"] = variant
            if best_state["first_success_variant"] is None and pred_label != target_label:
                best_state["first_success_variant"] = variant
                best_state["success_true_conf"] = var_true_conf
            if pred_label != target_label:
                print(f"[CODA] Attack Success at EST Phase! (Variant {idx})")
                return _build_result(
                    sample_id,
                    target_label,
                    original_code,
                    variant,
                    best_state["best_variant"],
                    best_state["first_success_variant"],
                    original_pred,
                    original_true_conf,
                    pred_label,
                    var_true_conf,
                    best_state["best_true_conf"],
                    best_state["success_true_conf"],
                    wrapper,
                )
            
            candidates_for_irt.append(variant)

        # --- Step 3: Identifier renaming transformation ---
        print(f"[CODA] Phase 3: IRT (Renaming) on {len(candidates_for_irt)} candidates...")
        
        for i, candidate_code in enumerate(candidates_for_irt):
            print(f"  > Running IRT on candidate {i+1}...")
            
            # Call the irt_phase implemented above.
            # Note: irt_phase already includes similarity computation and the replacement-attempt loop.
            flag, final_code, final_pred, final_true_conf = self._irt_phase(
                candidate_code, reference_paths, wrapper, target_label, best_state
            )
            final_variant = final_code
            
            if flag:
                print(f"[CODA] Attack Success at IRT Phase! (Based on EST Variant {i})")
                return _build_result(
                    sample_id,
                    target_label,
                    original_code,
                    final_code,
                    best_state["best_variant"],
                    best_state["first_success_variant"],
                    original_pred,
                    original_true_conf,
                    final_pred,
                    final_true_conf,
                    best_state["best_true_conf"],
                    best_state["success_true_conf"],
                    wrapper,
                )
        
        print("[CODA] Attack Failed. All phases exhausted.")
        final_data = src2embedding(final_variant, target_label)
        final_pred, final_true_conf = wrapper.predict_label_and_true_conf(final_data, target_label)
        if final_true_conf < best_state["best_true_conf"]:
            best_state["best_true_conf"] = final_true_conf
            best_state["best_variant"] = final_variant
        return _build_result(
            sample_id,
            target_label,
            original_code,
            final_variant,
            best_state["best_variant"],
            best_state["first_success_variant"],
            original_pred,
            original_true_conf,
            final_pred,
            final_true_conf,
            best_state["best_true_conf"],
            best_state["success_true_conf"],
            wrapper,
        )


def run_coda_attack(
    model_name,
    checkpoint_path,
    source_code,
    true_label,
    sample_id="unknown",
    sample_i=None,
    lang='cpp',
    input_dim=100,
    output_dim=200,
    codebert_path=LOCAL_CODEBERT_PATH,
    code_db_path=CODA_CODE_DB,
    src_paths_path=CODA_SRC_PATHS,
    fasttext_path='{HOME_PATH}/vul_robustness/CODA/data/saved_models/fasttext/ori_universal_fasttext.model',
    device='cuda',
    dataset_name=None,
):
    if dataset_name is not None:
        code_db_path, src_paths_path = get_coda_db_paths(model_name, dataset_name)
    effective_sample_id = sample_id if sample_id != "unknown" else (
        sample_i if sample_i is not None else "unknown"
    )
    # attacker = CODAAttacker(
    #     codebert_path=codebert_path,
    #     code_db_path=code_db_path,
    #     src_paths_path=src_paths_path,
    #     fasttext_path=fasttext_path,
    #     device=device,
    #     lang=lang,
    # )
    # ── Optimization: use the cache so all samples for the same victim share one instance ─────────────
    attacker = _get_or_create_attacker(
        codebert_path=codebert_path,
        code_db_path=code_db_path,
        src_paths_path=src_paths_path,
        fasttext_path=fasttext_path,
        device=device,
        lang=lang,
    )
    wrapper = ModelWrapper(model_name, checkpoint_path, input_dim=input_dim, output_dim=output_dim)
    return attacker.attack(
        target_code=source_code,
        wrapper=wrapper,
        target_label=true_label,
        sample_id=effective_sample_id,
    )

def release_coda_attacker(
    model_name: str = None,
    dataset_name: str = None,
    code_db_path: str = None,
    lang: str = "cpp",
    device: str = "cuda",
    codebert_path: str = LOCAL_CODEBERT_PATH,
    src_paths_path: str = CODA_SRC_PATHS,
    fasttext_path: str = '{HOME_PATH}/vul_robustness/CODA/data/saved_models/fasttext/ori_universal_fasttext.model',
):
    """
    Proactively release the CODAAttacker cache for a specified victim and clear GPU memory.
    Call this when switching victims so only the current victim's instance stays in GPU memory.
    """
    # Derive the key from model_name + dataset_name, matching run_coda_attack logic.
    if model_name is not None and dataset_name is not None:
        _code_db_path, _src_paths_path = get_coda_db_paths(model_name, dataset_name)
    else:
        _code_db_path = code_db_path or CODA_CODE_DB
        _src_paths_path = src_paths_path

    cache_key = (_code_db_path, _src_paths_path, codebert_path, fasttext_path, device, lang)

    with _ATTACKER_CACHE_LOCK:
        if cache_key in _ATTACKER_CACHE:
            del _ATTACKER_CACHE[cache_key]
            print(f"[CODA] Attacker cache released: {_code_db_path}")

    # Clear GPU fragments after release.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # 1. Initialize the attacker and victim model.
    # 2. Prepare data.
    sample_code = """
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
    """

    sample_code = """
void FillRandom() {
    ACMRandom rnd(ACMRandom::DeterministicSeed());
    for (int p = 0; p < num_planes_; p++) {
        for (int x = -1 ; x <= block_size_; x++)
            data_ptr_[p][x - stride_] = rnd.Rand8();
        for (int y = 0; y < block_size_; y++)
            data_ptr_[p][y * stride_ - 1] = rnd.Rand8();
    }
 }
 """
    # ref_paths = attacker._select_reference_inputs(sample_code, N=10)

    # est_candidates = attacker._est_transform(sample_code, ref_paths)

    # for idx, code in enumerate(est_candidates):
    #     print(f"\n=== EST Variant {idx+1} ===") 
    #     print(est_candidates)
    
    # # 3. Launch the attack.
    result = run_coda_attack(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        source_code=sample_code,
        true_label=0,
        sample_id="demo_coda",
        lang='cpp',
        input_dim=100,
        output_dim=200,
    )
    print(result.to_dict())