import hashlib
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
import os
from transformers import AutoTokenizer, AutoModel
from common.attack_result import AttackResult, to_text
from common.config_loader import get_settings
from common.utils.parser import initialize_language_parser, extract_identifiers_from_one_src, is_not_keyword
from common.utils.gen_embedding import src2embedding, src2pdg, pdg2embedding, load_word_vectors, renamed_pdg_to_embedding, multi_renamed_pdg_to_embedding
from src.model.wrapper import ModelWrapper
import threading

_ATTACKER_CACHE: dict = {}
_ATTACKER_CACHE_LOCK = threading.Lock()


def _get_or_create_attacker(
    codebert_path: str,
    code_db_path: str,
    src_paths_path: str,
    device: str,
    lang: str,
) -> "DIPAttacker":
    cache_key = (codebert_path, code_db_path, src_paths_path, device, lang)

    if cache_key in _ATTACKER_CACHE:
        return _ATTACKER_CACHE[cache_key]

    with _ATTACKER_CACHE_LOCK:
        if cache_key not in _ATTACKER_CACHE:
            print(f"[DIP] Creating new DIPAttacker for key: ({code_db_path}, lang={lang})")
            _ATTACKER_CACHE[cache_key] = DIPAttacker(
                codebert_path=codebert_path,
                code_db_path=code_db_path,
                src_paths_path=src_paths_path,
                device=device,
                lang=lang,
            )
    return _ATTACKER_CACHE[cache_key]

S = get_settings()
LOCAL_CODEBERT_PATH = S.local_codebert_path
DIP_CODE_DB = S.dip_code_db
DIP_SRC_PATHS = S.dip_src_paths
MODEL_NAME = "reveal"
CHECKPOINT_PATH = S.reveal_checkpoint


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
        attack_name="dip",
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

class DIPAttacker:
    def __init__(self, 
                 codebert_path=LOCAL_CODEBERT_PATH, 
                 code_db_path=DIP_CODE_DB, 
                 src_paths_path=DIP_SRC_PATHS, 
                 device='cuda',
                 lang='cpp'):
        """
        Main DIP attacker class: integrates vulnerable-position awareness, perturbation material selection, adversarial sample generation, and injection.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.lang = lang
        
        # 1. Load the proxy model (CodeBERT) for vector and attention computation.
        print(f"[DIP] Loading Proxy Model (CodeBERT) from {codebert_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_path)
        self.proxy_model = AutoModel.from_pretrained(codebert_path).to(self.device)
        self.proxy_model.eval()
        
        # 2. Initialize the Tree-sitter parser.
        print(f"[DIP] Initializing Tree-sitter Parser for {lang}...")
        self.parser = initialize_language_parser(lang)
        
        # 3. Load the dissimilar code database (Dissimilar Code DB).
        print(f"[DIP] Loading Dissimilar Code Database...")
        if os.path.exists(code_db_path) and os.path.exists(src_paths_path):
            # self.db_vecs = torch.load(code_db_path, map_location=self.device)
            self.db_vecs = torch.load(code_db_path, map_location="cpu")  # Keep resident on CPU.
            # Pre-normalize vectors to speed up later computations.
            self.db_vecs = F.normalize(self.db_vecs, p=2, dim=1)
            
            with open(src_paths_path, 'r', encoding='utf-8') as f:
                self.db_paths = json.load(f)
            print(f"[DIP] Database loaded. Size: {len(self.db_paths)}")
        else:
            print(f"[Error] DB files not found at {code_db_path} or {src_paths_path}")
            self.db_vecs = None
            self.db_paths = []

    # ================= Step 1: Vulnerable position awareness =================

    def _get_safe_insertion_lines(self, code_str):
        """Use Tree-sitter to parse code and return all syntactically safe insertion line numbers (1-based)."""
        tree = self.parser.parse(code_str.encode())
        insert_lines = set()

        def traverse(node):
            # Strategy: only focus on compound_statement ({...}).
            if node.type == 'compound_statement':
                # 1. At the beginning of a block, for example after '{'.
                if node.end_point[0] > node.start_point[0]:
                    insert_lines.add(node.start_point[0])
                
                # 2. After each child statement.
                for child in node.named_children:
                    # Ensure the line is still within the block range.
                    if child.end_point[0] < node.end_point[0]:
                        insert_lines.add(child.end_point[0])
                    traverse(child)
            else:
                for child in node.named_children:
                    traverse(child)

        traverse(tree.root_node)
        return sorted([line + 1 for line in insert_lines])

    def _get_code_vector(self, code_str):
        """Compute the CodeBERT [CLS] vector."""
        inputs = self.tokenizer(code_str, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.proxy_model(**inputs)
            return outputs.last_hidden_state[:, 0, :] # [1, 768]

    # def _rank_vulnerable_lines(self, code, valid_lines, top_k=5):
    #     """Evaluate vulnerability by inserting <unk> and computing vector displacement."""
    #     r_c = self._get_code_vector(code)
    #     candidates = []
    #     original_lines = code.split('\n')

    #     for line_num in valid_lines:
    #         # Construct perturbed code.
    #         temp_lines = original_lines.copy()
    #         if line_num <= len(temp_lines):
    #             temp_lines.insert(line_num, " <unk> ")
    #             perturbed_code = "\n".join(temp_lines)
                
    #             # Compute displacement.
    #             r_c_prime = self._get_code_vector(perturbed_code)
    #             cosine_sim = F.cosine_similarity(r_c, r_c_prime).item()
                
    #             # Higher scores indicate greater vulnerability (1 - sim).
    #             candidates.append({
    #                 "line": line_num,
    #                 "score": 1.0 - cosine_sim
    #             })

    #     candidates.sort(key=lambda x: x['score'], reverse=True)
    #     return candidates[:top_k]
    
    def _rank_vulnerable_lines(self, code, valid_lines, top_k=5, batch_size=8, target_vec=None):
        """
        Batched inference version: compose all perturbed code snippets into batches and send them to CodeBERT at once,
        replacing the original per-line inference approach.
        batch_size=32 can be adjusted based on GPU memory; reduce it when memory is tight.
        """
        # Reuse an externally provided vector when available; otherwise compute it here.
        r_c = target_vec.to(self.device) if target_vec is not None \
            else self._get_code_vector(code)
    
        original_lines = code.split('\n')

        # 1. Pre-generate all perturbed code snippets while recording their corresponding line numbers.
        perturbed_codes = []
        perturbed_line_nums = []
        for line_num in valid_lines:
            if line_num <= len(original_lines):
                temp_lines = original_lines.copy()
                temp_lines.insert(line_num, " <unk> ")
                perturbed_codes.append("\n".join(temp_lines))
                perturbed_line_nums.append(line_num)

        if not perturbed_codes:
            return []

        # 2. Compute the original code vector only once.
        r_c = self._get_code_vector(code)  # [1, 768]

        # 3. Tokenize and run forward propagation in batches to avoid exhausting GPU memory.
        all_vecs = []
        for i in range(0, len(perturbed_codes), batch_size):
            batch_codes = perturbed_codes[i : i + batch_size]
            inputs = self.tokenizer(
                batch_codes,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,          # Align sequences within the batch.
            ).to(self.device)
            with torch.no_grad():
                outputs = self.proxy_model(**inputs)
                batch_vecs = outputs.last_hidden_state[:, 0, :]  # [B, 768]
                all_vecs.append(batch_vecs)

        all_vecs = torch.cat(all_vecs, dim=0)  # [N, 768]

        # 4. Vectorize the computation of all cosine similarities in one pass.
        r_c_expanded = r_c.expand(all_vecs.size(0), -1)          # [N, 768]
        cosine_sims  = F.cosine_similarity(r_c_expanded, all_vecs, dim=1)  # [N]
        scores       = 1.0 - cosine_sims                          # Larger values indicate greater vulnerability.

        # 5. Construct and sort the results.
        candidates = [
            {"line": perturbed_line_nums[i], "score": scores[i].item()}
            for i in range(len(perturbed_line_nums))
        ]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    # ================= Step 2: Dissimilar code selection =================
    # def _select_dissimilar_code(self, target_code_str, sample_size=1000):
    #     """Select the code from the database that is most dissimilar to the target code."""
    #     if self.db_vecs is None: return None
        
    #     target_vec = self._get_code_vector(target_code_str)
    #     target_vec = F.normalize(target_vec, p=2, dim=1) # [1, 768]
        
    #     N = self.db_vecs.size(0)
        
    #     if sample_size and sample_size < N:
    #         # Random sampling (paper setting).
    #         indices = random.sample(range(N), sample_size)
    #         indices_tensor = torch.tensor(indices, device=self.device)
    #         candidate_vecs = self.db_vecs[indices_tensor]
            
    #         scores = torch.mm(candidate_vecs, target_vec.transpose(0, 1))
    #         min_val, min_idx_local = torch.min(scores, dim=0)
    #         best_global_idx = indices[min_idx_local.item()]
    #     else:
    #         # Full search.
    #         scores = torch.mm(self.db_vecs, target_vec.transpose(0, 1))
    #         best_global_idx = torch.argmin(scores).item()

    #     # Read the file.
    #     best_path = self.db_paths[best_global_idx]
    #     return self._read_file(best_path)
    def _select_dissimilar_code(self, target_code_str, sample_size=1000, target_vec=None):
        if self.db_vecs is None:
            return None

        # Use target_vec directly when provided externally; otherwise compute it here. Always move it to CPU.
        if target_vec is None:
            target_vec = self._get_code_vector(target_code_str).cpu()
        else:
            target_vec = target_vec.cpu()
        target_vec = F.normalize(target_vec, p=2, dim=1)  # [1, 768]，CPU

        N = self.db_vecs.size(0)  # db_vecs is already on CPU, so use it directly.

        if sample_size and sample_size < N:
             # Use a code hash as the seed so results are stable for the same code and varied across different code snippets.
            seed = int(hashlib.md5(target_code_str.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)          # Local Random instance; does not pollute the global seed.
            indices = rng.sample(range(N), sample_size)
            candidate_vecs = self.db_vecs[torch.tensor(indices)]   # CPU tensor
            scores = torch.mm(candidate_vecs, target_vec.T)
            best_global_idx = indices[torch.argmin(scores).item()]
        else:
            scores = torch.mm(self.db_vecs, target_vec.T)
            best_global_idx = torch.argmin(scores).item()

        return self._read_file(self.db_paths[best_global_idx])

    def _read_file(self, path):
        if not os.path.exists(path): return ""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except: return ""

    # ================= Step 3: Snippet extraction and dead-code generation =================

    def _extract_high_attention_snippet(self, code_str):
        """Extract the statement snippet with the highest attention score."""
        # 1. Tokenize and obtain offset_mapping.
        # Note: do not call .to(device) here because offset_mapping is not a tensor and would be hard to handle directly.
        inputs = self.tokenizer(code_str, return_tensors='pt', truncation=True, max_length=512, return_offsets_mapping=True)
        
        # 2. Key fix: extract offset_mapping and remove it from inputs.
        # pop() returns the value and removes the key from the dictionary, leaving only input_ids and attention_mask in inputs.
        offset_mapping = inputs.pop('offset_mapping') 
        
        # 3. Manually move the remaining tensors to device.
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # inputs no longer contains offset_mapping, so it can be safely passed in.
            outputs = self.proxy_model(**inputs, output_attentions=True)
            
        # Extract the CLS attention over all tokens.
        cls_attentions = outputs.attentions[-1][0, :, 0, :].mean(dim=0)
        
        # Find the strongest token.
        tokens = inputs['input_ids'][0]
        max_idx = -1
        max_score = -1
        
        # Iterate from 1 (skip <s>) to len-1 (skip </s>).
        for i in range(1, len(tokens)-1):
            score = cls_attentions[i].item()
            if score > max_score:
                max_score = score
                max_idx = i
                
        if max_idx == -1: return "int temp = 0;"
        
        # 4. Map back to code using the previously extracted offset_mapping.
        # offset_mapping is also a batched list; use [0].
        offset = offset_mapping[0][max_idx]
        start_char, end_char = offset[0].item(), offset[1].item()
        
        # Expand to a statement with Tree-sitter.
        try:
            tree = self.parser.parse(bytes(code_str, "utf8"))
            cursor = tree.root_node.descendant_for_byte_range(start_char, end_char)
            
            target_node = cursor
            while target_node:
                if "statement" in target_node.type or "declaration" in target_node.type:
                    break
                target_node = target_node.parent
                
            if target_node:
                return code_str[target_node.start_byte:target_node.end_byte]
            else:
                # Fallback: extract the containing line.
                return code_str.split('\n')[code_str[:start_char].count('\n')].strip()
        except Exception as e:
            # Prevent tree-sitter parsing failures on garbled text and provide a fallback.
            print(f"[Warning] Tree-sitter extract failed: {e}")
            return "int temp = 0;"

    def _wrap_as_dead_code(self, snippet, target_code_context):
        """Wrap the snippet as a char* variable."""
        # Mimic the variable-name style.
        if not isinstance(target_code_context, bytes):
            target_code_context = bytes(target_code_context, "utf8")
        existing_vars = extract_identifiers_from_one_src(target_code_context, lang=self.lang)
        if existing_vars:
            from collections import Counter
            base_name = Counter(existing_vars).most_common(1)[0][0]
            if len(base_name) < 2 or not is_not_keyword(base_name):
                base_name = "temp_str"
        else:
            base_name = "feature_str"
            
        var_name = f"{base_name}_{random.randint(100, 999)}"
        clean_snippet = snippet.replace('"', '\\"').replace('\n', ' ').strip()
        if len(clean_snippet) > 200: clean_snippet = clean_snippet[:200] + "..."
        
        return f'char* {var_name} = "{clean_snippet}";'

    # ================= Main attack interface =================

    def attack(self, target_code, wrapper: ModelWrapper, target_label, max_positions=8, sample_id="unknown"):
        """
        Execute the DIP attack.
        
        Args:
            target_code (str): Original code.
            target_label (int): Original correct label (Ground Truth).
            victim_model (nn.Module): Target model being attacked, such as CodeBERT or GraphCodeBERT.
            max_positions (int): Maximum number of vulnerable positions to try.
            
        Returns:
            AttackResult: Unified attack result object.
        """
        wrapper.reset_query_count()
        target_code = to_text(target_code)
        original_code = target_code
        last_variant = target_code

        ori_data = src2embedding(target_code.encode('utf-8'), target_label).to(torch.device(self.device))
        original_pred, original_true_conf = wrapper.predict_label_and_true_conf(ori_data, target_label)
        best_variant = target_code
        best_true_conf = original_true_conf
        first_success_variant = None
        success_true_conf = None
        if original_pred != target_label:
            return _build_result(
                sample_id,
                target_label,
                original_code,
                target_code,
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

        # 1. Find safe insertion points.
        safe_lines = self._get_safe_insertion_lines(target_code)
        if not safe_lines:
            print("[DIP] No safe insertion lines found.")
            return _build_result(
                sample_id,
                target_label,
                original_code,
                target_code,
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
        # ── Added: compute the target_code vector only once ──────────────────────────
        target_vec_cpu = self._get_code_vector(target_code).cpu()

        # 2. Vulnerability ranking: pass in the precomputed vector so it is not recomputed internally.
        ranked_candidates = self._rank_vulnerable_lines(
            target_code, safe_lines, top_k=max_positions,
            target_vec=target_vec_cpu,          # ← Added.
        )
        print(f"[DIP] Ranked Vulnerable Lines: {[item['line'] for item in ranked_candidates]}")

        # 3. Dissimilar code selection: reuse the same vector.
        dissimilar_code = self._select_dissimilar_code(
            target_code,
            target_vec=target_vec_cpu,          # ← Added.
        )    
        print(f"[DIP] Selected Dissimilar Code Snippet(200):\n{dissimilar_code[:200]}...\n")
        if not dissimilar_code:
            print("[DIP] Failed to select dissimilar code.")
            return _build_result(
                sample_id,
                target_label,
                original_code,
                target_code,
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
            
        # 4. Generate dead code.
        snippet = self._extract_high_attention_snippet(dissimilar_code)
        dead_code_stmt = self._wrap_as_dead_code(snippet, target_code)
        print(f"[DIP] Dead Code: {dead_code_stmt}")
        
        # 5. Iterative insertion and attack loop.
        if isinstance(target_code, str):
            ori_src_bytes = target_code.encode('utf-8')
        else:
            ori_src_bytes = target_code


        print(f"[DIP] Original Prediction: {original_pred} | True Label: {target_label}")

        lines = target_code.split('\n')
        
        for item in ranked_candidates:
            line_idx = item['line'] # 1-based, insert AFTER this line
            
            if line_idx >= len(lines): continue
            
            new_lines = lines[:]
            # Simple indentation handling; default is 4 spaces and can be optimized based on the previous line.
            indent = "    " 
            if line_idx > 0:
                prev_line = lines[line_idx-1]
                match = re.match(r'^(\s*)', prev_line)
                if match: indent = match.group(1)

            new_lines.insert(line_idx, f"{indent}{dead_code_stmt}")
            adv_code = "\n".join(new_lines)
            last_variant = adv_code
            print(f"[DIP] Trying insertion at line {line_idx}...")
            print(f"[DIP] {adv_code}") ## test
            
            # --- Query the victim model ---
            # Note: this assumes the victim model also accepts raw code strings and uses a similar tokenizer.
            # If the victim model requires special preprocessing, such as graph construction, adjust it here.
            
            adv_data = src2embedding(adv_code.encode('utf-8'), target_label).to(torch.device(self.device))
            pred_label, adv_true_conf = wrapper.predict_label_and_true_conf(adv_data, target_label)
            if adv_true_conf < best_true_conf:
                best_true_conf = adv_true_conf
                best_variant = adv_code
            if first_success_variant is None and pred_label != target_label:
                first_success_variant = adv_code
                success_true_conf = adv_true_conf
            if pred_label != target_label:
                print(f"[DIP] Success! Line {line_idx} | Original: {target_label} -> Adv: {pred_label}")
                return _build_result(
                    sample_id,
                    target_label,
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
                
        print("[DIP] Attack Failed.")
        final_data = src2embedding(last_variant.encode("utf-8"), target_label).to(torch.device(self.device))
        final_pred, final_true_conf = wrapper.predict_label_and_true_conf(final_data, target_label)
        if final_true_conf < best_true_conf:
            best_true_conf = final_true_conf
            best_variant = last_variant
        return _build_result(
            sample_id,
            target_label,
            original_code,
            last_variant,
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


def run_dip_attack(
    model_name,
    checkpoint_path,
    source_code,
    true_label,
    sample_id="unknown",
    sample_i=None,
    max_positions=8,
    lang='cpp',
    input_dim=100,
    output_dim=200,
    codebert_path=LOCAL_CODEBERT_PATH,
    code_db_path=DIP_CODE_DB,
    src_paths_path=DIP_SRC_PATHS,
    device='cuda',
):
    effective_sample_id = sample_id if sample_id != "unknown" else (
        sample_i if sample_i is not None else "unknown"
    )
    # attacker = DIPAttacker(
    #     codebert_path=codebert_path,
    #     code_db_path=code_db_path,
    #     src_paths_path=src_paths_path,
    #     device=device,
    #     lang=lang,
    # )
    # Optimization: prefer loading from cache.
    attacker = _get_or_create_attacker(
        codebert_path=codebert_path,
        code_db_path=code_db_path,
        src_paths_path=src_paths_path,
        device=device,
        lang=lang,
    )
    wrapper = ModelWrapper(model_name, checkpoint_path, input_dim=input_dim, output_dim=output_dim)
    return attacker.attack(
        target_code=source_code,
        wrapper=wrapper,
        target_label=true_label,
        max_positions=max_positions,
        sample_id=effective_sample_id,
    )


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
    true_label = 1 # Vulnerable.
    
    # 3. Launch the attack.
    result = run_dip_attack(
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        source_code=sample_code,
        true_label=true_label,
        sample_id="demo_dip",
        max_positions=3,
        lang='cpp',
        input_dim=100,
        output_dim=200,
    )
    print(result.to_dict())