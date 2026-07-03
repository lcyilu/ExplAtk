import os
import glob
import json
import functools
import re
from pathlib import Path
from multiprocessing import Pool, Manager, cpu_count
from typing import List, cast
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from src.utils.dot_parser import LooseDotParser

# Import FastText.
from gensim.models import FastText

# ==========================================
# 1. Preserve the original tokenization and processing logic.
# ==========================================

def tokenize_code_line(line):
    # Sets for operators. The original logic is preserved.
    operators3 = {'<<=', '>>='}
    operators2 = {
        '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
        '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
    }
    operators1 = {
        '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
        '=', ',', '?', ':', ';', '{', '}', '!', '~'
    }

    tmp, w = [], []
    i = 0
    if i is None: # Fix the type(i) == None style.
        return []
    while i < len(line):
        if line[i] == ' ':
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        elif line[i:i + 3] in operators3:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 3].strip())
            w = []
            i += 3
        elif line[i:i + 2] in operators2:
            tmp.append(''.join(w).strip())
            tmp.append(line[i:i + 2].strip())
            w = []
            i += 2
        elif line[i] in operators1:
            tmp.append(''.join(w).strip())
            tmp.append(line[i].strip())
            w = []
            i += 1
        else:
            w.append(line[i])
            i += 1
    if (len(w) != 0):
        tmp.append(''.join(w).strip())
        w = []
    tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
    return tmp

def process_parallel(path: str, split_token: bool):
    """
    Fully preserve the DOT parsing logic.
    Note: make sure LooseDotParser is already defined in your environment or can be imported.
    """
    tokens_list = list()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Assume LooseDotParser is available in your context.
        # from wherever import LooseDotParser 
        parser = LooseDotParser() 
        pdg = parser.to_networkx(content)

        for index, node in enumerate(pdg.nodes()):
                try:
                    label = pdg.nodes[node]['label'][1:-1]
                except:
                    continue
                code = label.partition(',')[2]
                for token in tokenize_code_line(code):
                    tokens_list.append(token)
    except Exception as e:
        # print(f"\n[Error Parsing File]: {path} - {e}")
        pass

    return tokens_list

# ==========================================
# 2. Modified FastText training function.
# ==========================================

def train_fasttext_embedding_from_ds(config_path: str, is_normalized: bool):
    """
    Train embeddings with FastText while fully reusing the existing dataset-path logic.
    """
    # 1. Path configuration. The original logic is preserved.
    if is_normalized:
        pdg_dirname = 'normal-pdg'
        # Change the save-path suffix to .model or .bin to make the format explicit.
        model_save_path = '{HOME_PATH}/vul_robustness/CODA/data/saved_models/fasttext/normal_universal_fasttext.model'
    else:
        pdg_dirname = 'ori-pdg'
        model_save_path = '{HOME_PATH}/vul_robustness/CODA/data/saved_models/fasttext/ori_universal_fasttext.model'
    
    devign_path = '{HOME_PATH}/VulDS/Devign'
    bigvul_path = '{HOME_PATH}/VulDS/BigVul'
    reveal_path = '{HOME_PATH}/VulDS/Reveal'

    devign_json = os.path.join(devign_path, 'devign_ds.json')
    bigvul_json = os.path.join(bigvul_path, 'bigvul_ds.json')
    reveal_json = os.path.join(reveal_path, 'reveal_ds.json')
    cwe119_json = os.path.join(bigvul_path, 'cwe119_ds.json')

    ds_json_list = [devign_json, bigvul_json, reveal_json, cwe119_json]
    dot_paths = set()

    # 2. Collect all DOT file paths. The original logic is preserved.
    print("Collecting DOT paths...")
    for ds_json in ds_json_list:
        try:
            with open(ds_json, 'r') as f:
                ds = json.load(f)
            ds_dir = os.path.dirname(ds_json)
            pdg_dir = os.path.join(ds_dir, pdg_dirname)

            for stem_type, stems in ds.items():
                # Normalize the path handling slightly; some dataset structures may differ while preserving the original behavior.
                train_path = Path(os.path.join(pdg_dir, stem_type[:-6]))
                if not train_path.exists():
                     # Fault-tolerant handling to avoid path-concatenation errors.
                     train_path = Path(os.path.join(pdg_dir, stem_type))
                
                stem_paths = train_path.glob('*.dot')
                # Keep only files defined in the JSON file.
                valid_stems = set(stems)
                stem_paths = [p for p in stem_paths if p.stem in valid_stems]
                
                # print(f"{train_path} total files: {len(stem_paths)}")
                dot_paths.update([str(p) for p in stem_paths])
        except Exception as e:
            print(f"Error processing {ds_json}: {e}")

    print(f"Total unique DOT files found: {len(dot_paths)}")

    # 3. Load the configuration.
    config = cast(DictConfig, OmegaConf.load(config_path))
    
    # 4. Extract tokens in parallel. The original logic is preserved.
    print("Extracting tokens using multiprocessing...")
    with Manager():
        pool = Pool(8) # Or config.num_workers.
        process_func = functools.partial(process_parallel, split_token=config.split_token)
        
        tokens: List = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, list(dot_paths)),
                desc=f"Processing PDGs",
                total=len(dot_paths),
            )
        ]
        pool.close()
        pool.join()

    print(f"Training FastText on {len(tokens)} documents...")
    
    # 5. Train the FastText model.
    num_workers = cpu_count() if config.num_workers == -1 else config.num_workers
    
    # Get parameter configuration.
    vector_size = getattr(config.gnn, 'embed_size', 128)
    vocab_size = getattr(config.dataset.token, 'vocabulary_size', None)
    ft_config = getattr(config, "fasttext", {})
    min_n = ft_config.get("min_n", 3)
    max_n = ft_config.get("max_n", 6)
    window = ft_config.get("window", 5)
    epochs = ft_config.get("epochs", 10)
    min_count = ft_config.get("min_count", 3)

    print(f"Training FastText with: size={vector_size}, min_n={min_n}, max_n={max_n}")

    model = FastText(
        sentences=tokens, 
        min_count=min_count,
        seed=64, 
        vector_size=vector_size,
        max_vocab_size=vocab_size,
        workers=num_workers, 
        sg=1,              # Skip-gram
        window=window,
        min_n=min_n,       # Use the value from the configuration file.
        max_n=max_n,       # Use the value from the configuration file.
        epochs=epochs
    )
    
    # 6. Save the model.
    # Note: FastText recommends saving the full model (.model), not only KeyedVectors (.wv).
    # The full model contains subword information, which is essential for computing OOV similarity.
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path) 
    
    # If you also need to save an old-format .wv copy, you can do so, but it will lose subword capability.
    # model.wv.save(model_save_path.replace('.model', '.wv'))
    print("Done.")

# ==========================================
# 3. Loading function. Updated accordingly.
# ==========================================
def load_fasttext_model(path: str):
    """
    Load the trained FastText model.
    """
    print(f"Loading FastText model from {path}...")
    # Use load instead of KeyedVectors.load to preserve OOV computation capability.
    model = FastText.load(path)
    return model

# ==========================================
# 4. Main Execution Entry Point
# ==========================================

def create_temp_config(path):
    """
    Create a temporary YAML configuration file to satisfy OmegaConf.load requirements.
    If you already have a real configuration file, use your own path directly and ignore this function.
    """
    config_content = """
dataset:
  name: "BigVul"
  token:
    vocabulary_size: 500000  # Vocabulary size
gnn:
  embed_size: 128            # Vector dimension. FastText defaults to 100; this script uses 128.
data_folder: "{HOME_PATH}/VulDS"
split_token: false
num_workers: 8               # Number of parallel processes
"""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(config_content)
        print(f"[Config] Created temporary config at {path}")
    else:
        print(f"[Config] Using existing config at {path}")

def test_model(model_path):
    """
    Simple test function to verify that the model was trained successfully and can handle OOV words.
    """
    print(f"\n>>> Testing Model: {model_path}")
    if not os.path.exists(model_path):
        print("Model file not found!")
        return

    # Load the model.
    model = FastText.load(model_path)
    print("Model loaded successfully.")

    # 1. Test a common word.
    test_word = "int"
    if test_word in model.wv:
        print(f"['{test_word}'] first 5 vector values: {model.wv[test_word][:5]}")
    else:
        print(f"['{test_word}'] is not in the vocabulary. It may have been filtered by min_count or the dataset may be too small.")

    # 2. Test FastText's core capability: OOV words.
    # Construct a compound word that is unlikely to appear directly in code but consists of common roots.
    oov_word = "fileReaderBufferTemp" 
    
    try:
        # Word2Vec would raise a KeyError here, but FastText should be able to generate a vector.
        vector = model.wv[oov_word]
        print(f"['{oov_word}'] (OOV) vector generated successfully! This is FastText's subword capability.")
        
        # Test similarity.
        sim = model.wv.similarity('file', oov_word)
        print(f"Sim('file', '{oov_word}') = {sim:.4f}")
        
    except KeyError:
        print(f"['{oov_word}'] vector generation failed. Please check the min_n/max_n parameters.")


def main():
    # 1. Configuration file path.
    config_path = "{HOME_PATH}/vul_robustness/CODA/src/fasttext.yaml"

    # 2. Start training.
    # is_normalized=True  -> use normal-pdg.
    # is_normalized=False -> use ori-pdg.
    print("\n>>> Start Training (Ori PDG)...")
    train_fasttext_embedding_from_ds(config_path, is_normalized=False)

    # 4. Verify model performance.
    # Corresponds to the save_path configured in train_fasttext_embedding_from_ds.
    save_path = '{HOME_PATH}/vul_robustness/CODA/data/saved_models/fasttext/ori_universal_fasttext.model'
    test_model(save_path)

if __name__ == "__main__":
    # Ensure multiprocessing works correctly on Linux.
    # Some libraries may have issues in fork mode. spawn is usually safer, but fork is typically the default in scripts.
    main()