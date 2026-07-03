import json
import os
from pathlib import Path
import re
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from src.config import DB_NAME, LOCAL_CODEBERT_PATH, TRAINING_SET_LIST_PATH,TRAINING_SET_SRC_PATH, MODEL_NAME, CHECKPOINT_PATH
from src.model.wrapper import ModelWrapper
from src.utils.gen_embedding import read_json
from src.utils.parser import initialize_language_parser, is_not_keyword, src2tree
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer,AutoTokenizer, AutoModel

class CodeBertTokenizerAligned:
    def __init__(self, model_name=LOCAL_CODEBERT_PATH, lang='cpp'):
        print(f"Loading tokenizer: {model_name} ...")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        
        # Load the C++ parser (make sure tree_sitter and the corresponding language library are installed)
        # This assumes that initialize_language_parser returns a tree_sitter.Parser object
        self.parser = initialize_language_parser(lang)

    def get_all_leaf_nodes(self, node):
        """Recursively retrieve all leaf nodes (retain the original filtering logic, but do not use it during <unk> replacement to avoid losing string literals and similar tokens)."""
        # Note: To preserve complete code reconstruction, string_literal nodes should generally not be skipped during <unk> replacement,
        # otherwise the reconstructed code may lose content and become syntactically invalid.
        # Keep the original function unchanged, but use a custom traversal in the function below.
        if node.type in ['comment', 'preproc_include']: # Skip comments only
            return []
        
        if len(node.children) == 0:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self.get_all_leaf_nodes(child))
        return leaves

    def replace_identifiers_with_unk(self, src):
        """
        Parse the source code into a syntax tree and traverse all nodes.
        Replace the text of nodes identified as identifiers with <unk>,
        while preserving the original spaces, line breaks, and operators.
        """
        if isinstance(src, str):
            src_bytes = src.encode('utf-8')
        else:
            src_bytes = src
            
        tree = self.parser.parse(src_bytes)
        root_node = tree.root_node
        
        # Use a custom traversal to ensure that no syntax token is omitted (such as strings or numbers)
        # Only comments are skipped, or they can be retained if needed
        leaves = []
        
        def traverse(node):
            # A node without children is a leaf node (token)
            if len(node.children) == 0:
                leaves.append(node)
            else:
                for child in node.children:
                    traverse(child)
        
        traverse(root_node)
        
        reconstructed_parts = []
        last_end_byte = 0
        
        for node in leaves:
            # 1. Process gaps between nodes (preserve spaces, line breaks, and indentation)
            start_byte = node.start_byte
            if start_byte > last_end_byte:
                # Retrieve the original content between two tokens (spaces, line breaks, etc.)
                gap = src_bytes[last_end_byte:start_byte].decode('utf-8', errors='replace')
                reconstructed_parts.append(gap)
                
            # 2. Retrieve the current node text
            node_text = src_bytes[start_byte:node.end_byte].decode('utf-8', errors='replace')
            
            # 3. Determine whether the node is a replacement target (identifier)
            is_target = (node.type in ['identifier', 'field_identifier', 'type_identifier']) and \
                        re.match(r'^[a-zA-Z_]\w*$', node_text) is not None

            # Exclude identifiers that may be language keywords 
            if  is_target and is_not_keyword(node_text):
                reconstructed_parts.append("<unk>")
            else:
                # Keep keywords, operators, literals (numbers/strings), semicolons, and similar tokens unchanged
                reconstructed_parts.append(node_text)
                
            last_end_byte = node.end_byte
            
        # Join the reconstructed parts
        return "".join(reconstructed_parts)


def generate_code_db(path_list_file=TRAINING_SET_LIST_PATH, batch_size=32, device='cuda', lang='c', model_name=MODEL_NAME, checkpoint_path=CHECKPOINT_PATH, db_name=DB_NAME):
    """
    1. Read source code -> 2. Replace identifiers with <unk> -> 3. Compute CLS vectors -> 4. Normalize and save
    """
    
    # --- A. Preparation ---
    output_vec_path = f"{HOME_PATH}/vul_robustness/CODA/data/raw/{model_name}_{db_name}_cls.pt"
    output_path_json = f"{HOME_PATH}/vul_robustness/CODA/data/raw/{model_name}_{db_name}_src_paths.json"
    
    print(f"Reading file list: {path_list_file}")
    with open(path_list_file, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]
        paths = sample_filter(paths, ModelWrapper(model_name, checkpoint_path))  # Filter out samples correctly predicted by the model
    total_files = len(paths)
    print(f"Total files to process: {total_files}")

    # --- B. Load the model and parser ---
    print(f"Loading CodeBERT and Tree-sitter ({lang})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_CODEBERT_PATH)
        model = AutoModel.from_pretrained(LOCAL_CODEBERT_PATH).to(device)
        model.eval()
        
        # Initialize the custom aligned tokenizer for <unk> processing
        unk_processor = CodeBertTokenizerAligned(model_name=LOCAL_CODEBERT_PATH, lang=lang)
        
    except Exception as e:
        print("Initialization failed:", e)
        return

    all_vectors = []
    valid_paths = []

    print("Starting batch processing (Abstraction + Encoding)...")
    
    # --- C. Batch processing ---
    for i in tqdm(range(0, total_files, batch_size)):
        batch_path_chunk = paths[i : i + batch_size]
        
        batch_unk_codes = []  # Store code in which identifiers have already been replaced with <unk>
        batch_paths_current = []

        # C-1. Read files and perform <unk> replacement
        for p in batch_path_chunk:
            if not os.path.exists(p): continue
            try:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    raw_content = f.read()
                
                # *** Key integration point: call unk_processor ***
                # Convert source code into a form such as "int <unk> = 0;"
                processed_content = unk_processor.replace_identifiers_with_unk(raw_content)
                
                if processed_content.strip(): # Ensure the processed content is not empty
                    batch_unk_codes.append(processed_content)
                    batch_paths_current.append(p)
                    
            except Exception as e:
                # print(f"File Error {p}: {e}")
                pass
        
        if not batch_unk_codes:
            continue

        # C-2. CodeBERT vectorization
        try:
            # The input text already contains <unk> replacements
            inputs = tokenizer(batch_unk_codes, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # Extract the CLS representation (the token at index 0)
                cls_vecs = outputs.last_hidden_state[:, 0, :]
                all_vectors.append(cls_vecs.cpu()) # Store on the CPU
                
                valid_paths.extend(batch_paths_current)
                
        except Exception as e:
            print(f"\n[Error] Batch {i} inference failed: {e}")
            continue

    # --- D. Post-processing and saving ---
    if not all_vectors:
        print("No vectors were generated.")
        return

    print("Concatenating and normalizing vectors...")
    final_tensor = torch.cat(all_vectors, dim=0)
    
    # L2 normalization (makes the dot product equivalent to cosine similarity)
    final_tensor = F.normalize(final_tensor, p=2, dim=1)

    print(f"Final vector matrix shape: {final_tensor.shape}")
    print(f"Number of valid paths: {len(valid_paths)}")

    # Ensure that the output directory exists
    os.makedirs(os.path.dirname(output_vec_path), exist_ok=True)

    print(f"Saving tensor to {output_vec_path} ...")
    torch.save(final_tensor, output_vec_path)
    
    print(f"Saving path list to {output_path_json} ...")
    with open(output_path_json, "w", encoding='utf-8') as f:
        json.dump(valid_paths, f, indent=2)

    print("Processing completed.")


def sample_filter(paths, wrapper:ModelWrapper):
    src_paths = []
    print("Collecting all misclassified training samples...")
    for p in tqdm(paths):
        data = read_json(p)
        if data is None:
            continue
        pred_label = wrapper.predict_label(data)
        true_label = int(wrapper.true_label(data))

        if pred_label == true_label:
            continue 
        
        path_obj = Path(p)
        subdir = path_obj.parent.name
        stem = path_obj.stem
        src_path = Path(TRAINING_SET_SRC_PATH) / subdir / (stem + ".c")
        src_paths.append(str(src_path))
    print(f"Selected {len(src_paths)} samples")
    return src_paths


def init_mlm(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = RobertaForMaskedLM.from_pretrained(LOCAL_CODEBERT_PATH)
    model.eval()
    model.to(device)

    return model
    

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

#     code = b'''
# int FUN1(char* VAR1) {
#     char VAR2[32];
#     char VAR3[] = "";
#     int VAR4 = 0;
#     int VAR5;


#     VAR5 = strlen(VAR1);
#     FUN2("", VAR5);


#     if (VAR5 > 0) {
#         VAR4 = 1;
#     }


#     strcpy(VAR2, VAR1);


#     FUN2("", VAR2);


#     if (strlen(VAR2) > 0) {
#         FUN2("");
#         return VAR4;
#     }

#     return -1;
# }                                                                                                             
# '''

    # nor_code = aligner.replace_identifiers_with_unk(code)   
    # print(f"\n[Code normalization] Code after replacing identifiers:\n{nor_code}\n")

    generate_code_db(path_list_file=TRAINING_SET_LIST_PATH, batch_size=16, device=torch.device("cuda:0"), lang='cpp', model_name=MODEL_NAME, checkpoint_path=CHECKPOINT_PATH)