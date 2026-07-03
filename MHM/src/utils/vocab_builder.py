import sys
from pathlib import Path

_R = Path(__file__).resolve().parents[3]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))

from src.config import MAX_VOCAB_SIZE, MIN_FREQ, VOCAB_SAVE_PATH
from common.utils.vocab_builder import (
    build_global_vocab as _build_global_vocab,
    build_global_vocab_from_dirs as _build_global_vocab_from_dirs,
)


def build_global_vacab_from_list(file_list, lang="cpp", save_name=None):
    return _build_global_vocab_from_dirs(
        file_list,
        max_vocab_size=MAX_VOCAB_SIZE,
        vocab_save_path=VOCAB_SAVE_PATH,
        lang=lang,
        save_name=save_name,
        min_freq=MIN_FREQ,
        output_style="dict_list",
    )


def build_global_vocab(source_dir, lang="cpp", save_name=None):
    return _build_global_vocab(
        source_dir,
        max_vocab_size=MAX_VOCAB_SIZE,
        vocab_save_path=VOCAB_SAVE_PATH,
        lang=lang,
        save_name=save_name,
        min_freq=MIN_FREQ,
    )


if __name__ == "__main__":
    # # Build vocabulary for a single dataset
    # source_directory = "{HOME_PATH}/VulDS/BigVul/normal_all_src"
    # build_global_vocab(source_directory, lang="cpp", save_name="normal_bigvul_vocab.json")

    # Build vocabularies for all datasets
    ori_src_list = [
        "{HOME_PATH}/VulDS/BigVul/all-src",
        "{HOME_PATH}/VulDS/Reveal/src",
        "{HOME_PATH}/VulDS/Devign/src"
    ]

    normal_src_list = [
        "{HOME_PATH}/VulDS/BigVul/normal-src",
        "{HOME_PATH}/VulDS/Reveal/normal-src",
        "{HOME_PATH}/VulDS/Devign/normal-src"
    ]

    build_global_vacab_from_list(ori_src_list,lang="cpp",save_name="ori_src_vocab.json")
    build_global_vacab_from_list(normal_src_list,lang="cpp",save_name="normal_src_vocab.json")