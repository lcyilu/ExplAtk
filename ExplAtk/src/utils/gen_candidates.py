import re
import threading
from gensim.models import KeyedVectors, Word2Vec
import torch
import transformers
from src.config import LOCAL_CODEBERT_PATH, WORD2VEC_PATH, LOCAL_CODET5_PATH
from src.utils.parser import initialize_language_parser, is_not_keyword, src2tree
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer, T5ForConditionalGeneration


# ════════════════════════════════════════════════════════════════
# Thread-safe process-level singleton cache
# ════════════════════════════════════════════════════════════════

# Cache the CodeBERT model by (path, device); all threads share the same GPU weights
_MLM_MODEL_CACHE = {}
_MLM_MODEL_LOCK = threading.Lock()

# Use thread-local caching for CodeBertTokenizerAligned:
# - Avoid reloading the tokenizer and rebuilding the tree-sitter parser on every gen_candis_* call
# - Use thread-local storage to avoid potential RobertaTokenizer / tree-sitter state issues in multithreading
#   One aligner per thread; four threads mean four aligners (tens of MB of CPU memory each, no GPU usage, negligible)
_aligner_local = threading.local()

class CodeBertTokenizerAligned:
    def __init__(self, model_name=LOCAL_CODEBERT_PATH, lang='cpp'):
        print(f"Loading tokenizer: {model_name} ...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Load the C++ parser
        self.parser = initialize_language_parser(lang)

    def get_all_leaf_nodes(self, node):
        """Recursively collect all leaf nodes."""
        # Blocklist for node types to skip entirely (comments, strings)
        if node.type in ['comment', 'string_literal', 'char_literal', 'preproc_include']:
            return []
        
        if len(node.children) == 0:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self.get_all_leaf_nodes(child))
        return leaves

    def tokenize_with_alignment(self, src):
        """
        Core function:
        Input: source-code string
        Output: 
          1. model_tokens: token list used as CodeBERT input (for example ['<s>', 'int', 'Ġs', 'ush', 'u', ...])
          2. alignment_map: a list with the same length as model_tokens.
             map[i] = {
                'source_text': 'sushu',   # which source-code word this token belongs to
                'node_type': 'identifier',# syntax type in the source code
                'is_target': True/False   # whether this is a potential attack target (variable name)
             }
        """
        # 1. Ensure the input is bytes for precise tree-sitter slicing
        if isinstance(src, str):
            src = src.encode('utf-8')
        tree = self.parser.parse(src)
        root_node = tree.root_node
        
        # 2. Get all leaf nodes (lexical tokens)
        leaf_nodes = self.get_all_leaf_nodes(root_node)
        
        # 3. Initialize result containers
        full_tokens = [self.tokenizer.cls_token] # [<s>]
        alignment_map = [None] # <s> has no corresponding source code
        
        last_end_byte = 0
        
        for node in leaf_nodes:
            # --- A. Get node text ---
            node_text = src[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
            
            # --- B. Handle whitespace before the node (important!) ---
            # CodeBERT (RoBERTa) relies on 'Ġ' (space) to distinguish word boundaries.
            # Check whether there is a gap between the current node and the previous node
            has_space_prefix = False
            if node.start_byte > last_end_byte:
                gap = src[last_end_byte:node.start_byte].decode('utf-8', errors='ignore')
                if len(gap) > 0 and gap.isspace():
                    has_space_prefix = True
            
            # Build tokenizer input: if there is preceding whitespace, RoBERTa needs a leading space before the word
            # Note: if the input string starts with a space, the RoBERTa tokenizer marks the first token as 'Ġ...'
            # Control this manually to simulate sentence continuity
            input_text = node_text
            if has_space_prefix:
                input_text = " " + node_text 
            
            # --- C. Call the tokenizer ---
            # add_special_tokens=False: we manually control <s> and </s>
            sub_tokens = self.tokenizer.tokenize(input_text)
            
            # Note: if this is the first word in the file, Ġ is usually not added even if there is no leading whitespace (tokenizer-dependent; RoBERTa is tricky)
            # The simplest approach is to trust the tokenizer's handling of " text".
            
            # --- D. Record the alignment mapping ---
            # Determine whether this is an attack target (identifier)
            # Simple logic: the type is identifier and the text follows variable naming rules
            is_target = (node.type == 'identifier' or node.type == 'field_identifier') and \
                        re.match(r'^[a-zA-Z_]\w*$', node_text) is not None
            
            # Map every generated sub-token to the current source node
            for sub_token in sub_tokens:
                full_tokens.append(sub_token)
                alignment_map.append({
                    'source_text': node_text,
                    'node_type': node.type,
                    'is_target': is_target,
                    'start_byte': node.start_byte, # useful for later source-code replacement
                    'end_byte': node.end_byte
                })
            
            last_end_byte = node.end_byte

        # 4. End-of-sequence handling
        full_tokens.append(self.tokenizer.sep_token) # [</s>]
        alignment_map.append(None)
        
        return full_tokens, alignment_map
    
def generate_candidates_for_variable(model, tokenizer, tokens, align_map, target_var_name, top_k=30):
    """
    Generate replacement candidates for the specified variable name (target_var_name).
    Strategy: replace all tokens of this variable with a *single* <mask>.
    """
    # 1. Find all index spans belonging to target_var_name
    # Structure: [ [2, 3, 4], [10, 11, 12] ]
    occurrences = [] 
    current_span = []
    
    for i, info in enumerate(align_map):
        if info and info['source_text'] == target_var_name and info['is_target']:
            current_span.append(i)
        else:
            if current_span:
                occurrences.append(current_span)
                current_span = []
    # Handle the trailing span
    if current_span: occurrences.append(current_span)
    
    if not occurrences:
        return []

    # 2. Build masked token IDs
    # Build a new token list by replacing each span with a single mask
    masked_token_ids = []
    token_ids_raw = tokenizer.convert_tokens_to_ids(tokens) # convert string tokens to int IDs
    mask_token_id = tokenizer.mask_token_id
    
    i = 0
    mask_indices_in_new_list = [] # record mask positions in the new list for retrieving predictions
    
    while i < len(token_ids_raw):
        # Check whether the current index is the start of a span
        is_start_of_span = False
        span_len = 0
        
        for span in occurrences:
            if span[0] == i:
                is_start_of_span = True
                span_len = len(span)
                break
        
        if is_start_of_span:
            # This is the start of a variable; insert one mask
            masked_token_ids.append(mask_token_id)
            mask_indices_in_new_list.append(len(masked_token_ids) - 1)
            # Skip all original tokens of this variable
            i += span_len
        else:
            # Regular token; copy as-is
            masked_token_ids.append(token_ids_raw[i])
            i += 1
    model_max_len = 512

    target_mask_idx = mask_indices_in_new_list[0] # use the first mask position

    print("Start Predicting Words!")
    # 3. Model prediction
    if len(masked_token_ids) > model_max_len:
        # Calculate the start and end positions of the window
        # Try to place the mask near the center of the window
        half_window = model_max_len // 2
        start = max(0, target_mask_idx - half_window)
        end = min(len(masked_token_ids), start + model_max_len)
        
        # Adjust start: if end reaches the tail, move start backward to keep the window length at 512
        if end - start < model_max_len:
            start = max(0, end - model_max_len)
            
        # Slice the window
        window_input_ids = masked_token_ids[start:end]
        
        # Adjust the mask index within the new window
        relative_mask_idx = target_mask_idx - start
        
        # Build the tensor
        # 1. Input: feed the entire window to provide context
        input_tensor = torch.tensor([window_input_ids]).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            # predictions shape: [Batch=1, Seq_Len=512, Vocab_Size=50265]
            predictions = outputs.logits 
            
        # 2. Extract: only use the prediction at the mask position
        # Even if the model outputs predictions for 512 positions, only take the row at relative_mask_idx
        target_token_logits = predictions[0, relative_mask_idx] 
        
        # 3. Sort and take Top-K
        top_k_probs, top_k_ids = torch.topk(target_token_logits, top_k)
    else:
        input_tensor = torch.tensor([masked_token_ids]).to(model.device)
    
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs.logits # [1, seq_len, vocab_size]
            
        # 4. Extract candidate words
        # Usually aggregate predictions across all mask positions (for example, use the first mask prediction or the intersection/product of all mask predictions)
        # ALERT's simple approach: use only the first mask prediction (the context is bidirectional, so the model knows all masks refer to the same variable)
        
        probs = predictions[0, target_mask_idx] # [vocab_size]
        
        top_k_probs, top_k_ids = torch.topk(probs, top_k)
    
    results = []
    for idx in top_k_ids:
        word = tokenizer.decode([idx]).strip()
        # Simple filtering: remove the original name and special characters
        if word != target_var_name and word.isidentifier() and is_not_keyword(word):
        # if word != target_var_name and word.isidentifier():
            results.append(word)
    return results

def generate_candidates_for_variable_codet5(model, tokenizer, tokens, align_map, target_var_name, top_k=30):
    """
    Generate replacement candidates for the specified variable name (target_var_name) using CodeT5.
    Strategy:
    1. Replace all tokens of this variable with <extra_id_0>.
    2. Use model.generate to generate top_k sequences.
    3. Parse the sequences to extract words.
    """
    
    # ==========================
    # 1. Find all positions of the variable (same logic as CodeBERT)
    # ==========================
    occurrences = [] 
    current_span = []
    
    for i, info in enumerate(align_map):
        if info and info['source_text'] == target_var_name and info['is_target']:
            current_span.append(i)
        else:
            if current_span:
                occurrences.append(current_span)
                current_span = []
    if current_span: occurrences.append(current_span)
    
    if not occurrences:
        return []

    # ==========================
    # 2. Build masked input IDs
    # ==========================
    masked_token_ids = []
    token_ids_raw = tokenizer.convert_tokens_to_ids(tokens)
    
    # CodeT5 sentinel ID (sentinel token)
    # For codet5-base, the ID of <extra_id_0> is usually 32099
    # If the tokenizer loads special tokens, tokenizer.convert_tokens_to_ids('<extra_id_0>') can also be used
    # For robustness, try to retrieve it first; if retrieval fails, use the default value
    sentinel_id = tokenizer.convert_tokens_to_ids('<extra_id_0>')
    if sentinel_id == tokenizer.unk_token_id:
        sentinel_id = 32099 
    
    i = 0
    mask_indices_in_new_list = [] 
    
    while i < len(token_ids_raw):
        is_start_of_span = False
        span_len = 0
        
        for span in occurrences:
            if span[0] == i:
                is_start_of_span = True
                span_len = len(span)
                break
        
        if is_start_of_span:
            # Difference here: CodeT5 replaces the entire span with *one* sentinel
            masked_token_ids.append(sentinel_id)
            mask_indices_in_new_list.append(len(masked_token_ids) - 1)
            i += span_len
        else:
            masked_token_ids.append(token_ids_raw[i])
            i += 1
            
    # ==========================
    # 3. Windowing
    # ==========================
    model_max_len = 512
    target_mask_idx = mask_indices_in_new_list[0] # center the window around the first mask

    final_input_ids = []

    if len(masked_token_ids) > model_max_len:
        half_window = model_max_len // 2
        start = max(0, target_mask_idx - half_window)
        end = min(len(masked_token_ids), start + model_max_len)
        
        if end - start < model_max_len:
            start = max(0, end - model_max_len)
            
        final_input_ids = masked_token_ids[start:end]
    else:
        final_input_ids = masked_token_ids

    # Convert to tensor
    input_tensor = torch.tensor([final_input_ids]).to(model.device)

    # ==========================
    # 4. CodeT5 generation (core difference)
    # ==========================
    print(f"CodeT5 Predicting for: {target_var_name}...")
    
    # Use beam search to generate multiple results
    outputs = model.generate(
        input_tensor, 
        max_length=16,             # variable names are usually short, so long generation is unnecessary
        num_beams=top_k + 5,       # beam width is slightly larger than k to improve diversity
        num_return_sequences=top_k,# return k sequences
        early_stopping=True
    )
    
    # ==========================
    # 5. Parsing and filtering
    # ==========================
    candidates = []
    seen_candidates = set() # deduplication
    
    for output_ids in outputs:
        # Decode
        raw_text = tokenizer.decode(output_ids, skip_special_tokens=False)
        # print(f"Raw Generated Text: {raw_text}")
        
        # Parse: the standard format is "<pad> <extra_id_0> prediction <extra_id_1> </s>"
        # It may also be "<pad> <s> <extra_id_0> prediction <extra_id_1> </s>"
        text_content = raw_text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
        
        predicted_word = ""
        
        if "<extra_id_0>" in text_content:
            parts = text_content.split("<extra_id_0>")
            if len(parts) > 1:
                content_after = parts[1]
                if "<extra_id_1>" in content_after:
                    predicted_word = content_after.split("<extra_id_1>")[0].strip()
                else:
                    predicted_word = content_after.strip()
            preds = re.findall(r'[a-zA-Z_]\w*', predicted_word)
            predicted_word = preds[0] if preds else predicted_word

        # If parsing fails, the output may be plain text; use it directly
        if not predicted_word:
             predicted_word = text_content.replace("<extra_id_0>", "").strip()

        # --- Filtering logic ---
        if not predicted_word: continue
        
        # 1. Remove whitespace (CodeT5 may generate spaces with a 'Ġ'-like effect; after decoding they are normal spaces)
        predicted_word = predicted_word.strip()
        
        # 2. Exclude the original name, non-identifiers, keywords, and duplicates
        if predicted_word != target_var_name and predicted_word.isidentifier() and is_not_keyword(predicted_word) and predicted_word not in seen_candidates:
            seen_candidates.add(predicted_word)
            candidates.append(predicted_word)
        
            
        if len(candidates) >= top_k:
            break
            
    return candidates

def init_mlm(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Thread-safe version with a process-level singleton cache.
    All threads share one CodeBERT instance, so GPU memory stores only one copy (previously four threads meant four copies).
    The signature matches the original version, so upstream {atk_name}.py does not need any changes.
    """
    key = (LOCAL_CODEBERT_PATH, str(device))

    # Fast path: lock-free read
    cached = _MLM_MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    # Slow path: double-checked locking; zero overhead after the first access
    with _MLM_MODEL_LOCK:
        cached = _MLM_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        print(f"[init_mlm] Loading CodeBERT for the first time: {LOCAL_CODEBERT_PATH} -> {device}")
        # print(f"[init_mlm] Loading CodeBERT for the first time: {LOCAL_CODET5_PATH} -> {device}")
        model = RobertaForMaskedLM.from_pretrained(LOCAL_CODEBERT_PATH)
        # model = T5ForConditionalGeneration.from_pretrained(LOCAL_CODET5_PATH)
        model.eval()
        model.to(device)
        _MLM_MODEL_CACHE[key] = model
        return model
    
def _get_aligner(model_name, lang='cpp'):
    """
    Thread-local CodeBertTokenizerAligned cache.
    Each thread builds one aligner on its first call and reuses it, avoiding repeated
    RobertaTokenizer loading (vocab/merges files and repeated tens-of-MB I/O) on every gen_candis_* call.
    """
    cache = getattr(_aligner_local, 'cache', None)
    if cache is None:
        cache = {}
        _aligner_local.cache = cache
    key = (model_name, lang)
    if key in cache:
        return cache[key]
    aligner = CodeBertTokenizerAligned(model_name=model_name, lang=lang)
    cache[key] = aligner
    return aligner

def gen_candis(code, mlm_model, target_var, *, _precomputed=None):
    """
    Generate CodeBERT candidate words.
    
    Parameters
    ----------
    _precomputed : tuple | None
        If (tokens, align_map, tokenizer) is provided, skip the repeated tokenize step.
        It can be obtained via precompute_tokenize().
    """
    if _precomputed is not None:
        tokens, align_map, tokenizer = _precomputed
    else:
        aligner = _get_aligner(LOCAL_CODEBERT_PATH)
        tokens, align_map = aligner.tokenize_with_alignment(code)
        tokenizer = aligner.tokenizer

    candidates = generate_candidates_for_variable(model=mlm_model, tokenizer=tokenizer, tokens=tokens, align_map=align_map, target_var_name=target_var)

    return candidates

def gen_candis_codet5(code, mlm_model, target_var, *, _precomputed=None):
    """
    Generate CodeT5 candidate words.
    
    Parameters
    ----------
    _precomputed : tuple | None
        If (tokens, align_map, tokenizer) is provided, skip the repeated tokenize step.
        It can be obtained via precompute_tokenize_codet5().
    """
    if _precomputed is not None:
        tokens, align_map, tokenizer = _precomputed
    else:
        aligner = _get_aligner(LOCAL_CODET5_PATH)
        tokens, align_map = aligner.tokenize_with_alignment(code)
        tokenizer = aligner.tokenizer

    candidates = generate_candidates_for_variable_codet5(model=mlm_model, tokenizer=tokenizer, tokens=tokens, align_map=align_map, target_var_name=target_var)

    return candidates


def precompute_tokenize(code, lang='cpp'):
    """
    Tokenize the code once with CodeBERT and return the (tokens, align_map, tokenizer) tuple.
    When generating candidates for multiple variables in the same code later, pass this result to
    gen_candis(..., _precomputed=result) to avoid repeated tokenization.
    """
    aligner = _get_aligner(LOCAL_CODEBERT_PATH, lang=lang)
    tokens, align_map = aligner.tokenize_with_alignment(code)
    return tokens, align_map, aligner.tokenizer


def precompute_tokenize_codet5(code, lang='cpp'):
    """
    Tokenize the code once with CodeT5 and return the (tokens, align_map, tokenizer) tuple.
    When generating candidates for multiple variables in the same code later, pass this result to
    gen_candis_codet5(..., _precomputed=result) to avoid repeated tokenization.
    """
    aligner = _get_aligner(LOCAL_CODET5_PATH, lang=lang)
    tokens, align_map = aligner.tokenize_with_alignment(code)
    return tokens, align_map, aligner.tokenizer
    
import numpy as np

def most_dissimilar_w2v_fast(wv, target, top_k=10, exclude=None):
    """
    Find the top_k words in wv that are least similar to target.

    target can be:
      1. str: a token in the vocabulary
      2. np.ndarray/list: a precomputed vector, such as mean_vec

    Returns:
      List[(word, similarity)]
    """
    exclude = set(exclude or [])

    # case 1: target is a vocabulary token
    if isinstance(target, str):
        if target not in wv.key_to_index:
            return []

        target_vec = wv.get_vector(target, norm=True)
        exclude.add(target)

    # case 2: target is a vector, such as mean_vec
    else:
        target_vec = np.asarray(target, dtype=np.float32).reshape(-1)

        if target_vec.shape[0] != wv.vector_size:
            raise ValueError(
                f"target vector dim mismatch: got {target_vec.shape[0]}, "
                f"expected {wv.vector_size}"
            )

        norm = np.linalg.norm(target_vec)

        if norm == 0 or not np.isfinite(norm):
            return []

        target_vec = target_vec / norm

    # All word vectors, already normalized
    all_vecs = wv.get_normed_vectors()

    # cosine similarity
    sims = np.dot(all_vecs, target_vec)

    # Exclude existing variable names / the original word
    for word in exclude:
        idx = wv.key_to_index.get(word)
        if idx is not None:
            sims[idx] = np.inf

    k = min(top_k, len(sims))

    if k <= 0:
        return []

    # Take the k entries with the smallest similarity
    idx = np.argpartition(sims, k - 1)[:k]
    idx = idx[np.argsort(sims[idx])]

    return [
        (wv.index_to_key[i], float(sims[i]))
        for i in idx
    ]

def generate_candidates_w2v(
    wv,          # loaded gensim Word2Vec vocabulary
    target_var: str,    # target variable name, such as "sushu_counter"
    top_k: int = 5,    # number of candidate words to return
) -> list:
    """
    Generate candidate identifiers based on Word2Vec cosine similarity.

    Strategy:
      1. Directly check whether target_var is in the W2V vocabulary
      2. If it is not in the vocabulary (OOV), split underscore subwords and query using their mean vector
      3. Filter out the original word, non-identifiers, and C++ keywords
    """

    # ── Case 1: the word is directly in the vocabulary ─────────────────────────
    if target_var in wv:
        similar = wv.most_similar(target_var, topn=top_k * 2)  # fetch extra items so top_k remain after filtering
        # similar = most_dissimilar_w2v_fast(wv,target_var,top_k * 2)

    # ── Case 2: OOV; try the mean of subword vectors for snake_case variable names ──
    else:
        parts = [p for p in re.split(r'[_\d]+', target_var) if p and p in wv]
        if not parts:
            # Completely OOV; cannot process
            print(f"[W2V] '{target_var}' and its subwords are not in the vocabulary; returning an empty list")
            return []

        import numpy as np
        mean_vec = np.mean([wv[p] for p in parts], axis=0)
        similar  = wv.most_similar([mean_vec], topn=top_k * 2)
        # similar = most_dissimilar_w2v_fast(wv,mean_vec,top_k * 2)
        print(f"[W2V] '{target_var}' is OOV; querying with the mean vector of subwords {parts}")

    # ── Filtering ──────────────────────────────────────────────────────────────
    candidates = []
    for word, score in similar:
        if (
            word != target_var          # not the original word itself
            and word.isidentifier()     # valid identifier
            and is_not_keyword(word)    # not a keyword
        ):
            candidates.append(word)
        if len(candidates) >= top_k:
            break

    return candidates


# ── Replacement for the original gen_candis ────────────────────────────────
def gen_candis_w2v(
    target_var: str,
    wv,
    top_k: int = 30,
) -> list:
    """
    Direct replacement for the original gen_candis().
    code, mlm_model, and aligner are no longer needed, so the interface is simpler.
    """
    return generate_candidates_w2v(wv, target_var, top_k)
    

if __name__ == "__main__":
    aligner = CodeBertTokenizerAligned()
    
    code = "int sushu = 0; if (sushu > 10) { return; }"
    code = b"""
static void  rv34_pred_mv (RV34DecContext *r, int block_type, int subblock_no, int dmv_no) {
    MpegEncContext *s = &r->s;
    int mv_pos = s->mb_x * 2 + s->mb_y * 2 * s->b8_stride;
    int A [2] = {0}, B [2], C [2];
    int i, j;
    int mx, my;
    int avail_index = avail_indexes[subblock_no];
    int c_off = part_sizes_w[block_type];
    mv_pos += (subblock_no & 1) + (subblock_no >> 1) * s->b8_stride;
    if (subblock_no == 3)
        c_off = -1;
    if (r->avail_cache[avail_index - 1]) {
        A[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - 1][0];
        A[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - 1][1];
    }
    if (r->avail_cache[avail_index - 4]) {
        B[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride][0];
        B[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride][1];
    }
    else {
        B[0] = A[0];
        B[1] = A[1];
    }
    if (!r->avail_cache[avail_index - 4 + c_off]) {
        if (r->avail_cache[avail_index - 4] & &(r->avail_cache[avail_index - 1] || r->rv30)) {
            C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride - 1][0];
            C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride - 1][1];
        }
        else {
            C[0] = A[0];
            C[1] = A[1];
        }
    }
    else {
        C[0] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride + c_off][0];
        C[1] = s->current_picture_ptr->f.motion_val[0][mv_pos - s->b8_stride + c_off][1];
    }
    mx = mid_pred (A[0], B[0], C[0]);
    my = mid_pred (A[1], B[1], C[1]);
    mx += r->dmv[dmv_no][0];
    my += r->dmv[dmv_no][1];
    {
        j = 0;
        while (j < part_sizes_h[block_type]) {
            for (i = 0; i < part_sizes_w[block_type]; i++) {
                s->current_picture_ptr->f.motion_val[0][mv_pos + i + j * s->b8_stride][0] = mx;
                s->current_picture_ptr->f.motion_val[0][mv_pos + i + j * s->b8_stride][1] = my;
            }
            j++;
        }
    }
}
"""

    code = b'''
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

    # print(f"\nOriginal code: {code}\n")
    
    tokens, align_map = aligner.tokenize_with_alignment(code)
    
    # # Simulated attack: find all token indices for 'sushu'
    # attack_indices = []
    
    # for i, token in enumerate(tokens):
    #     info = align_map[i]
    #     source_text = info['source_text'] if info else "N/A"
    #     node_type = info['node_type'] if info else "N/A"
    #     is_target = info['is_target'] if info else False
        
    #     print(f"{i:<6} | {token:<12} | {source_text:<12} | {node_type:<15} | {is_target}")
        
    #     if source_text == 'sushu':
    #         attack_indices.append(i)

    # print(f"\n[Attack target localization] Token indices corresponding to variable 'sushu': {attack_indices}")
    # print("This means that if you want to mask 'sushu', you need to replace the IDs at these positions with <mask>.")

    # Load the model
    model = init_mlm()
    target_var = 'VAR1'
    candidates = generate_candidates_for_variable(model, aligner.tokenizer, tokens, align_map, target_var)

    print(f"CodeBERT recommended replacement words for variable '{target_var}': {candidates}")