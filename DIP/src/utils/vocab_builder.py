import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))

from src.config import MAX_VOCAB_SIZE, VOCAB_SAVE_PATH
from common.utils.vocab_builder import (
    build_global_vocab as _build_global_vocab,
    build_global_vacab_from_list as _build_global_vacab_from_list,
)


def build_global_vacab_from_list(file_list, lang="cpp", save_name=None):
    return _build_global_vacab_from_list(
        file_list,
        max_vocab_size=MAX_VOCAB_SIZE,
        vocab_save_path=VOCAB_SAVE_PATH,
        lang=lang,
        save_name=save_name,
        min_freq=None,
        output_style="tuple_set",
    )


def build_global_vocab(source_dir, lang="cpp", save_name=None):
    return _build_global_vocab(
        source_dir,
        max_vocab_size=MAX_VOCAB_SIZE,
        vocab_save_path=VOCAB_SAVE_PATH,
        lang=lang,
        save_name=save_name,
        min_freq=None,
    )


if __name__ == "__main__":
    source_directory = "{HOME_PATH}/VulDS/BigVul/normal_all_src"
    build_global_vocab(source_directory, lang="cpp", save_name="normal_bigvul_vocab.json")
