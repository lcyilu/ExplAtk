import re
import torch
import transformers
from src.config import LOCAL_CODEBERT_PATH
from src.utils.parser import initialize_language_parser, is_not_keyword, src2tree
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

class CodeBertTokenizerAligned:
    def __init__(self, model_name=LOCAL_CODEBERT_PATH, lang='cpp'):
        print(f"Loading tokenizer: {model_name} ...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Load the C++ parser.
        self.parser = initialize_language_parser(lang)

    def get_all_leaf_nodes(self, node):
        """Recursively get all leaf nodes."""
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
        Input: source code string.
        Output:
          1. model_tokens: token list used as CodeBERT input, for example ['<s>', 'int', 'Ġs', 'ush', 'u', ...].
          2. alignment_map: a list with the same length as model_tokens.
             map[i] = {
                'source_text': 'sushu',   # Which source-code word this token belongs to.
                'node_type': 'identifier',# Syntax type in the source code.
                'is_target': True/False   # Whether it is a potential attack target, such as a variable name.
             }
        """
        # 1. Ensure the input is bytes for precise tree-sitter slicing.
        if isinstance(src, str):
            src = src.encode('utf-8')
        tree = self.parser.parse(src)
        root_node = tree.root_node
        
        # 2. Get all leaf nodes (lexical tokens).
        leaf_nodes = self.get_all_leaf_nodes(root_node)
        
        # 3. Initialize result containers.
        full_tokens = [self.tokenizer.cls_token] # [<s>]
        alignment_map = [None] # <s> has no corresponding source code.
        
        last_end_byte = 0
        
        for node in leaf_nodes:
            # --- A. Get the node text. ---
            node_text = src[node.start_byte:node.end_byte].decode('utf-8', errors='replace')
            
            # --- B. Handle whitespace before the node (important). ---
            # CodeBERT (RoBERTa) relies on 'Ġ' (space) to distinguish word boundaries.
            # Check whether there is a gap between the current node and the previous node.
            has_space_prefix = False
            if node.start_byte > last_end_byte:
                gap = src[last_end_byte:node.start_byte].decode('utf-8', errors='ignore')
                if len(gap) > 0 and gap.isspace():
                    has_space_prefix = True
            
            # Build tokenizer input: if there is leading whitespace, RoBERTa needs a space before the word.
            # Note: RoBERTa marks the first token as 'Ġ...' when the input string starts with a space.
            # Manually control this to simulate sentence continuity.
            input_text = node_text
            if has_space_prefix:
                input_text = " " + node_text 
            
            # --- C. Call the tokenizer. ---
            # add_special_tokens=False: manually control <s> and </s>.
            sub_tokens = self.tokenizer.tokenize(input_text)
            
            # Fix: if this is the first word in the file, it usually should not receive Ġ even if there is no source-code space, depending on the tokenizer implementation.
            # The simplest approach is to rely directly on how the tokenizer handles " text".
            
            # --- D. Record the alignment mapping. ---
            # Determine whether this is an attack target (Identifier).
            # Simple logic: the type is identifier and the text follows variable-naming rules.
            is_target = (node.type == 'identifier' or node.type == 'field_identifier') and \
                        re.match(r'^[a-zA-Z_]\w*$', node_text) is not None
            
            # Point each generated sub-token to the current source node.
            for sub_token in sub_tokens:
                full_tokens.append(sub_token)
                alignment_map.append({
                    'source_text': node_text,
                    'node_type': node.type,
                    'is_target': is_target,
                    'start_byte': node.start_byte, # Useful for later source-code replacement.
                    'end_byte': node.end_byte
                })
            
            last_end_byte = node.end_byte

        # 4. Handle the ending token.
        full_tokens.append(self.tokenizer.sep_token) # [</s>]
        alignment_map.append(None)
        
        return full_tokens, alignment_map
    
def generate_candidates_for_variable(model, tokenizer, tokens, align_map, target_var_name, top_k=30):
    """
    Generate replacement candidates for the specified variable name (target_var_name).
    Strategy: replace all tokens of that variable with a *single* <mask>.
    """
    # 1. Find all index spans that belong to target_var_name.
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
    # Handle the ending span.
    if current_span: occurrences.append(current_span)
    
    if not occurrences:
        return []

    # 2. Build masked token IDs.
    # Build a new token list by replacing each span with a single mask.
    masked_token_ids = []
    token_ids_raw = tokenizer.convert_tokens_to_ids(tokens) # Convert string tokens to integer IDs.
    mask_token_id = tokenizer.mask_token_id
    
    i = 0
    mask_indices_in_new_list = [] # Record mask positions in the new list for extracting predictions.
    
    while i < len(token_ids_raw):
        # Check whether the current i is the start of a span.
        is_start_of_span = False
        span_len = 0
        
        for span in occurrences:
            if span[0] == i:
                is_start_of_span = True
                span_len = len(span)
                break
        
        if is_start_of_span:
            # This is the start of a variable, so insert one mask.
            masked_token_ids.append(mask_token_id)
            mask_indices_in_new_list.append(len(masked_token_ids) - 1)
            # Skip all original tokens for this variable.
            i += span_len
        else:
            # Regular token; copy it as-is.
            masked_token_ids.append(token_ids_raw[i])
            i += 1
    model_max_len = 512

    target_mask_idx = mask_indices_in_new_list[0] # Use the first mask position.

    print("Start Predicting Words!")
    # 3. Model prediction.
    if len(masked_token_ids) > model_max_len:
        # Compute the window start and end positions.
        # Try to keep the mask near the center of the window.
        half_window = model_max_len // 2
        start = max(0, target_mask_idx - half_window)
        end = min(len(masked_token_ids), start + model_max_len)
        
        # Adjust start: if end reaches the end, move start backward to keep the window length at 512.
        if end - start < model_max_len:
            start = max(0, end - model_max_len)
            
        # Slice the window.
        window_input_ids = masked_token_ids[start:end]
        
        # Adjust the mask index within the new window.
        relative_mask_idx = target_mask_idx - start
        
        # Build the tensor.
        # 1. Input: feed the full window to provide context.
        input_tensor = torch.tensor([window_input_ids]).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            # predictions shape: [Batch=1, Seq_Len=512, Vocab_Size=50265]
            predictions = outputs.logits 
            
        # 2. Extract: only inspect the prediction at the Mask position.
        # Even though the model outputs predictions for 512 positions, only the relative_mask_idx row is used.
        target_token_logits = predictions[0, relative_mask_idx] 
        
        # 3. Sort and take Top-K.
        top_k_probs, top_k_ids = torch.topk(target_token_logits, top_k)
    else:
        input_tensor = torch.tensor([masked_token_ids]).to(model.device)
    
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = outputs.logits # [1, seq_len, vocab_size]
            
        # 4. Extract candidate words.
        # Usually aggregate predictions from all mask positions, such as using the first mask prediction or the intersection/product of all mask predictions.
        # ALERT: simple approach: only use the first mask prediction, because the bidirectional context tells the model that all masks are the same variable.
        
        probs = predictions[0, target_mask_idx] # [vocab_size]
        
        top_k_probs, top_k_ids = torch.topk(probs, top_k)
    
    results = []
    for idx in top_k_ids:
        word = tokenizer.decode([idx]).strip()
        # Simple filtering: remove the original name and special characters.
        if word != target_var_name and word.isidentifier() and is_not_keyword(word):
        # if word != target_var_name and word.isidentifier():
            results.append(word)
    return results

def init_mlm(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = RobertaForMaskedLM.from_pretrained(LOCAL_CODEBERT_PATH)
    model.eval()
    model.to(device)

    return model

def gen_candis(code, mlm_model, target_var):
    aligner = CodeBertTokenizerAligned()
    tokens, align_map = aligner.tokenize_with_alignment(code)

    candidates = generate_candidates_for_variable(model=mlm_model, tokenizer=aligner.tokenizer, tokens=tokens, align_map=align_map, target_var_name=target_var)

    return candidates
    

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
    
    # print(f"{'Index':<6} | {'Token':<12} | {'Source Text':<12} | {'Type':<15} | {'Target?'}")
    # print("-" * 65)
    
    # # Simulated attack: find all token indices for 'sushu'.
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
    # print("This means that if you want to mask 'sushu', you need to replace all these positions with <mask>.")

    # Load the model.
    model = init_mlm()
    target_var = 'VAR1'
    candidates = generate_candidates_for_variable(model, aligner.tokenizer, tokens, align_map, target_var)

    print(f"CodeBERT recommended replacement words for variable '{target_var}': {candidates}")
