import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.config_loader import get_settings

_s = get_settings()

LANGUAGE_SO_PATH = _s.language_so_path
DEFAULT_LANGUAGE = _s.default_language

MAX_VOCAB_SIZE = _s.max_vocab_size_mhm
MIN_FREQ = _s.min_freq_mhm
VOCAB_SAVE_PATH = _s.vocab_dir

MHM_MAX_ITER = _s.mhm_max_iter

MODEL_NAME = "reveal"
CHECKPOINT_PATH = _s.reveal_checkpoint
DEVICE = _s.device
BATCH_SIZE = _s.batch_size_mhm

JOERN_PATH = _s.joern_path
WORD2VEC_PATH = _s.word2vec_path

MASK_PLACEHOLDER = _s.mask_placeholder
MASK = _s.mask
