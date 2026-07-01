"""Load global `config.yaml` from the project root (single source of truth for paths)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_CACHE: dict[str, Any] | None = None
_SETTINGS: AppSettings | None = None


def project_root() -> Path:
    return Path(load_config().get("project_root", _PROJECT_ROOT))


def _expand(value: Any, root: Path) -> Any:
    """Substitute only ``{project_root}`` so other placeholders (e.g. ``{db_name}``) stay intact."""
    if isinstance(value, str):
        return value.replace("{project_root}", str(root))
    if isinstance(value, dict):
        return {k: _expand(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, root) for v in value]
    return value


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Return merged config dict; paths are expanded with ``project_root``."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE
    path = config_path or (_PROJECT_ROOT / "config.yaml")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    root = Path(raw.get("project_root", _PROJECT_ROOT))
    expanded = _expand(raw, root)
    if config_path is None:
        _CONFIG_CACHE = expanded
    return expanded


def reload_config() -> None:
    """Clear caches (e.g. after editing config.yaml in tests)."""
    global _CONFIG_CACHE, _SETTINGS
    _CONFIG_CACHE = None
    _SETTINGS = None


@dataclass
class AppSettings:
    """Typed view over the global YAML config."""

    data: dict[str, Any]

    @property
    def project_root(self) -> Path:
        return Path(self.data["project_root"])

    @property
    def language_so_path(self) -> str:
        return self.data["paths"]["language_so"]

    @property
    def vocab_dir(self) -> str:
        return self.data["paths"]["vocab_dir"]

    @property
    def joern_path(self) -> str:
        return self.data["paths"]["joern"]

    @property
    def word2vec_path(self) -> str:
        return self.data["models"]["word2vec"]

    @property
    def reveal_checkpoint(self) -> str:
        return self.data["models"]["reveal_checkpoint"]

    @property
    def ivdetect_checkpoint(self) -> str:
        return self.data["models"]["ivdetect_checkpoint"]

    @property
    def local_codebert_path(self) -> str:
        return self.data["models"]["local_codebert"]

    @property
    def local_codet5_path(self) -> str:
        return self.data["models"]["local_codet5"]

    @property
    def dip_code_db(self) -> str:
        return self.data["datasets"]["dip_code_db"]

    @property
    def dip_src_paths(self) -> str:
        return self.data["datasets"]["dip_src_paths"]

    @property
    def coda_db_dir(self) -> str:
        return self.data["datasets"]["coda_db_dir"]

    @property
    def coda_code_db(self) -> str:
        return self.data["datasets"]["coda_code_db"]

    @property
    def coda_src_paths(self) -> str:
        return self.data["datasets"]["coda_src_paths"]

    @property
    def training_set_src_path(self) -> str:
        return self.data["datasets"]["training_set_src"]

    def training_set_list_path(self, db_name: str) -> str:
        tpl = self.data["datasets"]["training_set_list_template"]
        return tpl.format(db_name=db_name)

    @property
    def default_language(self) -> str:
        return self.data["language"]["default"]

    @property
    def max_vocab_size_default(self) -> int:
        return int(self.data["vocab"]["max_size_default"])

    @property
    def max_vocab_size_mhm(self) -> int:
        return int(self.data["vocab"]["max_size_mhm"])

    @property
    def min_freq_mhm(self) -> int:
        return int(self.data["vocab"]["min_freq_mhm"])

    @property
    def mhm_max_iter(self) -> int:
        return int(self.data["attack"]["mhm_max_iter"])

    @property
    def mask_placeholder(self) -> str:
        return self.data["attack"]["mask_placeholder"]

    @property
    def mask(self) -> str:
        return self.data["attack"]["mask"]

    @property
    def pop_size(self) -> int:
        return int(self.data["attack"]["pop_size"])

    @property
    def max_gen(self) -> int:
        return int(self.data["attack"]["max_gen"])

    @property
    def moaa_max_gen(self) -> int:
        return int(self.data["attack"]["moaa_max_gen"])

    @property
    def mutation_rate(self) -> float:
        return float(self.data["attack"]["mutation_rate"])

    @property
    def device(self) -> str:
        return self.data["device"]["default"]

    @property
    def batch_size_mhm(self) -> int:
        return int(self.data["training"]["batch_size_mhm"])


def get_settings() -> AppSettings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = AppSettings(load_config())
    return _SETTINGS
