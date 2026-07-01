from __future__ import annotations
import collections
import json
import os
from typing import Iterable

from tqdm import tqdm

from common.utils.parser import extract_identifiers_from_one_src


def build_global_vacab_from_list(
    file_list: Iterable[str],
    *,
    max_vocab_size: int,
    vocab_save_path: str,
    lang: str = "cpp",
    save_name: str | None = None,
    min_freq: int | None = None,
    output_style: str = "dict_list",
):
    """Build vocabulary from explicit file paths.

    ``output_style``: ``"dict_list"`` (MHM-style list of dicts) or ``"tuple_set"``
    (Alert-style set of (id, freq) pairs, serialized as list of pairs).
    """
    counter: collections.Counter[str] = collections.Counter()
    for file in tqdm(file_list, desc="Processing source files"):
        try:
            with open(file, "rb") as f:
                src_bytes = f.read()
            identifiers = extract_identifiers_from_one_src(src_bytes, lang)
            counter.update(identifiers)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    pairs = counter.most_common(max_vocab_size)
    if min_freq is not None:
        pairs = [(i, f) for i, f in pairs if f >= min_freq]
    if output_style == "tuple_set":
        global_vocab = {(identifier, freq) for identifier, freq in pairs}
        out = list(global_vocab)
    elif output_style == "dict_list":
        out = [{identifier: freq} for identifier, freq in pairs]
    else:
        raise ValueError("output_style must be 'dict_list' or 'tuple_set'")
    if save_name is None:
        vocab_json_path = os.path.join(vocab_save_path, "global_vocab.json")
    else:
        vocab_json_path = os.path.join(vocab_save_path, save_name)
    with open(vocab_json_path, "w") as f:
        json.dump(out, f, indent=4)


def build_global_vocab(
    source_dir: str,
    *,
    max_vocab_size: int,
    vocab_save_path: str,
    lang: str = "cpp",
    save_name: str | None = None,
    min_freq: int | None = None,
):
    counter: collections.Counter[str] = collections.Counter()
    print(f"Building vocabulary from source directory: {source_dir}")
    for root, _, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Processing files in {root}"):
            if file.endswith((".c", ".cpp", ".h", ".hpp")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "rb") as f:
                        src_bytes = f.read()
                    identifiers = extract_identifiers_from_one_src(src_bytes, lang)
                    counter.update(identifiers)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
    pairs = counter.most_common(max_vocab_size)
    if min_freq is not None:
        pairs = [(i, f) for i, f in pairs if f >= min_freq]
    global_vocab = [{identifier: freq} for identifier, freq in pairs]
    if save_name is None:
        vocab_json_path = os.path.join(vocab_save_path, "global_vocab.json")
    else:
        vocab_json_path = os.path.join(vocab_save_path, save_name)
    with open(vocab_json_path, "w") as f:
        json.dump(list(global_vocab), f, indent=4)

def build_global_vocab_from_dirs(
    source_dirs: Iterable[str],
    *,
    max_vocab_size: int,
    vocab_save_path: str,
    lang: str = "cpp",
    save_name: str | None = None,
    min_freq: int | None = None,
    output_style: str = "dict_list",
):
    """从多个目录递归收集文件，构建全局词表。"""
    
    # 递归收集所有目录下的目标文件
    all_files = [
        os.path.join(root, file)
        for source_dir in source_dirs
        for root, _, files in os.walk(source_dir)
        for file in files
        if file.endswith((".c", ".cpp", ".h", ".hpp"))
    ]
    print(f"共从 {len(list(source_dirs))} 个目录中找到 {len(all_files)} 个源文件")

    # 直接复用 from_list 版本
    build_global_vacab_from_list(
        file_list       = all_files,
        max_vocab_size  = max_vocab_size,
        vocab_save_path = vocab_save_path,
        lang            = lang,
        save_name       = save_name,
        min_freq        = min_freq,
        output_style    = output_style,
    )
