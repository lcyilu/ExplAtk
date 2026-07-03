import re
import threading
from gensim.models import KeyedVectors, Word2Vec
import torch
import transformers
from src.config import LOCAL_CODEBERT_PATH, WORD2VEC_PATH
from src.utils.parser import initialize_language_parser, is_not_keyword, src2tree
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

# ════════════════════════════════════════════════════════════════
# Thread-safe process-wide singleton cache
# ════════════════════════════════════════════════════════════════

# Cache the CodeBERT model by (path, device); all threads share the same GPU weights
_MLM_MODEL_CACHE = {}
_MLM_MODEL_LOCK = threading.Lock()

# Use thread-local cache for CodeBertTokenizerAligned:
# - Avoid reloading the tokenizer and rebuilding the tree-sitter parser on every gen_candis_* call
# - Thread-local storage helps avoid potential state issues in RobertaTokenizer / tree-sitter under multithreading
#   Each thread keeps one aligner; four threads mean four aligners (tens of MB of CPU memory each, no GPU memory, negligible)
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
          1. model_tokens: token list used as CodeBERT input, e.g., ['<s>', 'int', 'Ġs', 'ush', 'u', ...]
          2. alignment_map: a list with the same length as model_tokens.
             map[i] = {
                'source_text': 'sushu',   # The source-code word this token belongs to
                'node_type': 'identifier',# The syntax type in the source code
                'is_target': True/False   # Whether it is a potential attack target (variable name)
             }
        """
        # 1. Ensure bytes input for exact tree-sitter slicing
        if isinstance(src, str):
            src = src.encode('utf-8')
        tree = self.parser.parse(src)
        root_node = tree.root_node
        
        # 2. Collect all leaf nodes (lexical tokens)
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
            
            # Build the tokenizer input: if there is preceding whitespace, RoBERTa needs a leading space before the word
            # Note: if the input string starts with a space, the RoBERTa tokenizer marks the first token as 'Ġ...'
            # Manually control this to simulate sentence continuity
            input_text = node_text
            if has_space_prefix:
                input_text = " " + node_text 
            
            # --- C. Call the tokenizer ---
            # add_special_tokens=False: <s> and </s> are controlled manually
            sub_tokens = self.tokenizer.tokenize(input_text)
            
            # Note: if this is the first word in the file, Ġ is usually not added even without source whitespace
            # (the exact behavior depends on the tokenizer implementation; RoBERTa can be tricky).
            # The simplest approach is to rely on the tokenizer's handling of " text".
            
            # --- D. Record the alignment mapping ---
            # Determine whether this is an attack target (identifier)
            # Simple rule: the type is identifier and the text follows variable-naming rules
            is_target = (node.type == 'identifier' or node.type == 'field_identifier') and \
                        re.match(r'^[a-zA-Z_]\w*$', node_text) is not None
            
            # Map each generated sub-token to the current source node
            for sub_token in sub_tokens:
                full_tokens.append(sub_token)
                alignment_map.append({
                    'source_text': node_text,
                    'node_type': node.type,
                    'is_target': is_target,
                    'start_byte': node.start_byte, # Useful for subsequent source-code replacement
                    'end_byte': node.end_byte
                })
            
            last_end_byte = node.end_byte

        # 4. Finalization
        full_tokens.append(self.tokenizer.sep_token) # [</s>]
        alignment_map.append(None)
        
        return full_tokens, alignment_map
    
def generate_candidates_for_variable(model, tokenizer, tokens, align_map, target_var_name, top_k=30):
    """
    Generate replacement candidates for the specified variable name (target_var_name).
    Strategy: replace all tokens of the variable with a *single* <mask>.
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
    # Handle the final span
    if current_span: occurrences.append(current_span)
    
    if not occurrences:
        return []

    # 2. Build masked token IDs
    # Build a new token list where each span is replaced by a single mask
    masked_token_ids = []
    token_ids_raw = tokenizer.convert_tokens_to_ids(tokens) # Convert string tokens to integer IDs
    mask_token_id = tokenizer.mask_token_id
    
    i = 0
    mask_indices_in_new_list = [] # Record mask positions in the new list for later prediction lookup
    
    while i < len(token_ids_raw):
        # Check whether the current i is the start of a span
        is_start_of_span = False
        span_len = 0
        
        for span in occurrences:
            if span[0] == i:
                is_start_of_span = True
                span_len = len(span)
                break
        
        if is_start_of_span:
            # This is the start of a variable, so insert one mask
            masked_token_ids.append(mask_token_id)
            mask_indices_in_new_list.append(len(masked_token_ids) - 1)
            # Skip all original tokens of this variable
            i += span_len
        else:
            # Regular token; copy it directly
            masked_token_ids.append(token_ids_raw[i])
            i += 1
    model_max_len = 512

    target_mask_idx = mask_indices_in_new_list[0] # Use the position of the first mask

    print("Start Predicting Words!")
    # 3. Model prediction
    if len(masked_token_ids) > model_max_len:
        # Compute the window boundaries
        # Try to place the mask near the center of the window
        half_window = model_max_len // 2
        start = max(0, target_mask_idx - half_window)
        end = min(len(masked_token_ids), start + model_max_len)
        
        # Adjust start: if end reaches the sequence boundary, move start backward to keep a 512-token window
        if end - start < model_max_len:
            start = max(0, end - model_max_len)
            
        # Slice the window
        window_input_ids = masked_token_ids[start:end]
        
        # Adjust the mask index in the new window
        relative_mask_idx = target_mask_idx - start
        
        # Build the tensor
        # 1. Input: feed the entire window to provide context
        input_tensor = torch.tensor([window_input_ids]).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            # predictions shape: [Batch=1, Seq_Len=512, Vocab_Size=50265]
            predictions = outputs.logits 
            
        # 2. Extract: only use the prediction at the mask position
        # Even if the model outputs predictions for 512 positions, only the relative_mask_idx row is used
        target_token_logits = predictions[0, relative_mask_idx] 
        
        # 3. Sort and take Top-K
        top_k_probs, top_k_ids = torch.topk(target_token_logits, top_k)
    else:
        input_tensor = torch.tensor([masked_token_ids]).to(model.device)
    
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs.logits # [1, seq_len, vocab_size]
            
        # 4. Extract candidate words
        # Usually, predictions across all mask positions can be combined, e.g., using the first mask prediction,
        # or taking the intersection/product of predictions from all masks.
        # ALERT's simple approach only uses the prediction at the first mask because the context is bidirectional
        # and the model knows all masks refer to the same variable.
        
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

def init_mlm(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Thread-safe version with a process-wide singleton cache.
    All threads share the same CodeBERT model, so GPU memory is used only once
    instead of once per thread.
    The signature remains identical to the original version, so upstream moaa.py does not need any changes.
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
        model = RobertaForMaskedLM.from_pretrained(LOCAL_CODEBERT_PATH)
        model.eval()
        model.to(device)
        _MLM_MODEL_CACHE[key] = model
        return model

def _get_aligner(model_name, lang='cpp'):
    """
    Thread-local cache for CodeBertTokenizerAligned.
    Each thread constructs one aligner on its first call and reuses it, avoiding repeated
    RobertaTokenizer loading (vocab/merges files and tens of MB of repeated I/O) on every gen_candis_* call.
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


def gen_candis(code, mlm_model, target_var):
    aligner = _get_aligner(LOCAL_CODEBERT_PATH)
    tokens, align_map = aligner.tokenize_with_alignment(code)

    candidates = generate_candidates_for_variable(model=mlm_model, tokenizer=aligner.tokenizer, tokens=tokens, align_map=align_map, target_var_name=target_var)

    return candidates

def generate_candidates_w2v(
    wv,          # Loaded gensim Word2Vec vocabulary
    target_var: str,    # Target variable name, e.g., "sushu_counter"
    top_k: int = 5,    # Number of candidates to return
) -> list:
    """
    Generate candidate identifiers based on Word2Vec cosine similarity.

    Strategy:
      1. Directly query whether target_var exists in the W2V vocabulary
      2. If it is out-of-vocabulary (OOV), split underscore-separated subtokens and query with their mean vector
      3. Filter out the original word, non-identifiers, and C++ keywords
    """

    # ── Case 1: the word is directly in the vocabulary ─────────────────────
    if target_var in wv:
        similar = wv.most_similar(target_var, topn=top_k * 2)  # Retrieve more candidates so top_k remain after filtering

    # ── Case 2: OOV; try subtoken mean vectors for snake_case variable names ─
    else:
        parts = [p for p in re.split(r'[_\d]+', target_var) if p and p in wv]
        if not parts:
            # Completely OOV; cannot process
            print(f"[W2V] '{target_var}' and all of its subtokens are not in the vocabulary; returning an empty list")
            return []

        import numpy as np
        mean_vec = np.mean([wv[p] for p in parts], axis=0)
        similar  = wv.most_similar([mean_vec], topn=top_k * 2)
        print(f"[W2V] '{target_var}' is OOV; querying with the mean vector of subtokens {parts}")

    # ── Filtering ───────────────────────────────────────────────────────────
    candidates = []
    for word, score in similar:
        if (
            word != target_var          # Not the original word itself
            and word.isidentifier()     # Valid identifier
            and is_not_keyword(word)    # Not a keyword
        ):
            candidates.append(word)
        if len(candidates) >= top_k:
            break

    return candidates


# ── Replace the original gen_candis ─────────────────────────────────────────
def gen_candis_w2v(
    target_var: str,
    wv,
    top_k: int = 30,
) -> list:
    """
    Direct replacement for the original gen_candis().
    code, mlm_model, and aligner are no longer needed, making the interface simpler.
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
    
    # # Simulate an attack: find all token indices for 'sushu'
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
    # print("This means that to mask 'sushu', the IDs at these positions should be replaced with <mask>.")

    # Load the model
    model = init_mlm()
    target_var = 'VAR1'
    candidates = generate_candidates_for_variable(model, aligner.tokenizer, tokens, align_map, target_var)

    print(f"CodeBERT recommended replacement words for variable '{target_var}': {candidates}")