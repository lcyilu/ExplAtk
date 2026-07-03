import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from src.config import LOCAL_CODEBERT_PATH

def generate_code_db(path_list_file, batch_size=32, device='cuda'):
    """
    1. Read all source files.
    2. Compute CLS vectors in batches using CodeBERT.
    3. Apply global normalization.
    4. Save the vector Tensor and the corresponding path list.
    """
    
    # 1. Read the path list.
    print(f"Reading file list: {path_list_file}")
    with open(path_list_file, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]
    
    total_files = len(paths)
    print(f"Total files to process: {total_files}")

    # 2. Load CodeBERT.
    print(f"Loading CodeBERT (Device: {device})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_CODEBERT_PATH)
        model = AutoModel.from_pretrained(LOCAL_CODEBERT_PATH).to(device)
    except Exception as e:
        print("Failed to load the model. Please check the network connection or path.")
        raise e
    
    model.eval()

    all_vectors = []    # Temporary vector list.
    valid_paths = []    # Temporary list of successfully processed paths, aligned with vectors.

    # 3. Batch-processing loop.
    print("Starting batch vector computation...")
    
    # Use tqdm to display the progress bar.
    for i in tqdm(range(0, total_files, batch_size)):
        # Get the paths for the current batch.
        batch_path_chunk = paths[i : i + batch_size]
        
        batch_codes = []
        batch_paths_temp = []

        # Read code content.
        for p in batch_path_chunk:
            if not os.path.exists(p):
                continue
            try:
                # errors='replace' prevents read failures caused by special characters.
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    batch_codes.append(content)
                    batch_paths_temp.append(p)
            except Exception as e:
                # Skip files that fail to read to keep the data clean.
                pass
        
        if not batch_codes:
            continue

        # Tokenize & Inference
        try:
            # max_length=512 is CodeBERT's hard limit.
            inputs = tokenizer(batch_codes, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # Get CLS vectors (batch_size, 768).
                # last_hidden_state has shape [batch, seq_len, hidden].
                # [:, 0, :] selects the 0th token for all samples, i.e., <s>/CLS.
                cls_vecs = outputs.last_hidden_state[:, 0, :]
                
                # Move vectors back to CPU to avoid running out of GPU memory.
                all_vectors.append(cls_vecs.cpu())
                
                # Add paths to the final list only after inference succeeds.
                valid_paths.extend(batch_paths_temp)
                
        except Exception as e:
            print(f"\n[Error] Batch {i} failed: {e}")
            continue

    # 4. Aggregate and normalize.
    if not all_vectors:
        print("No vectors were generated. Please check whether the path file is correct.")
        return

    print("Merging and normalizing vectors...")
    # Merge the list into one large Tensor [N, 768].
    final_tensor = torch.cat(all_vectors, dim=0)

    # *** Key step: L2 normalization ***
    # After normalization, the dot product of two vectors equals cosine similarity.
    final_tensor = F.normalize(final_tensor, p=2, dim=1)

    print(f"Final vector shape: {final_tensor.shape}")
    print(f"Final number of valid paths: {len(valid_paths)}")

    # 5. Save files.
    print("Saving files...")
    torch.save(final_tensor, "{HOME_PATH}/vul_robustness/DIP/data/raw/code_db.pt")
    
    with open("{HOME_PATH}/vul_robustness/DIP/data/raw/src_paths.json", "w", encoding='utf-8') as f:
        json.dump(valid_paths, f, indent=2)

    print("Done. Generated 'code_db.pt' and 'src_paths.json'.")

if __name__ == "__main__":
    # Input file path.
    path_file = "{HOME_PATH}/vul_robustness/DIP/data/raw/all_unique_source_paths.txt"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Batch size.
    generate_code_db(path_file, batch_size=16, device=device)