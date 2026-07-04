# ExplAtk: Explanation-Guided Adversarial Attacks against GNN-Based Vulnerability Detectors

**ExplAtk** is an explanation-guided adversarial attack framework for GNN-based vulnerability detection models. It uses graph explanations to locate prediction-critical graph nodes, graph edges, source statements, and identifiers, then uses the projected source-level importance score `S(z)` to guide candidate generation and adversarial search.

This repository includes five baselines, **MHM**, **ALERT**, **DIP**, **CODA**, and **MOAA**, together with our method **ExplAtk**. All attacks follow the same entry convention and return the same `AttackResult` format.

```text
<method>/src/attack/<attack_method_name>.py
```

---

## 1. Environment Setup

Create the Python environment from the project root:

```bash
conda env create -f environment.yml
conda activate explatk
pip install -r requirements.txt
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```


### Joern

Joern is used to parse C/C++ source code and export graph representations. A reproducibility-oriented Joern version is included under:

```text
preprocess/joern-1.1.172/
```

Make sure the executables are available:

```bash
chmod +x preprocess/joern-1.1.172/joern
chmod +x preprocess/joern-1.1.172/joern-parse
chmod +x preprocess/joern-1.1.172/joern-export
```

The official Joern repository is:

```text
https://github.com/joernio/joern
```

Before running preprocessing, training, or attacks, update local paths in `config.yaml`, such as dataset paths, checkpoint paths, Joern paths, embedding paths, and output directories.

---

## 2. Victim Model Training and Preprocessing

Our victim-model setup is built on **Interpreters for GNN-based Vulnerability Detection: Are We There Yet?** by Hu et al. and its released implementation, **vdgraph**:

```text
https://github.com/CGCL-codes/vdgraph
```

The original work provides the datasets, victim model implementations, and the basic pipeline from source code to graph representation, Word2Vec training, graph embedding, and GNN model training. We follow this setup and add several preprocessing improvements to obtain stronger and more stable victim models.

```bibtex
@inproceedings{hu2023interpreters,
  title={Interpreters for GNN-based vulnerability detection: Are we there yet?},
  author={Hu, Yutao and Wang, Suyuan and Li, Wenke and Peng, Junru and Wu, Yueming and Zou, Deqing and Jin, Hai},
  booktitle={Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages={1407--1419},
  year={2023}
}
```

### Victim models and datasets

We train four GNN-based vulnerability detectors:

| Victim model | Description |
|---|---|
| Devign | Learns vulnerability patterns from graph representations of source code. |
| IVDetect | Uses program-dependence information and graph convolution. |
| Reveal | Learns vulnerability-related graph representations. |
| DeepWukong | Uses normalized code and vulnerability-oriented program slices. |

The models are trained on four datasets:

| Dataset | Construction | Train / Val / Test |
|---|---|---|
| Devign | Augmented from the original Devign dataset | 58,462 / 19,490 / 19,490 |
| BigVul | Augmented from the original BigVul dataset | 45,511 / 15,209 / 15,209 |
| CWE119 | Derived from augmented BigVul by selecting CWE-119 samples | 15,492 / 5,167 / 5,167 |
| Reveal | Augmented from the original Reveal dataset | 14,288 / 4,764 / 4,764 |

This gives 16 victim settings:

```text
4 victim models x 4 datasets = 16 victim settings
```

### Why additional preprocessing is needed

Training directly on the original BigVul, Devign, and Reveal datasets led to two issues:

1. The trained victim models sometimes had weak base performance.
2. Some datasets were highly imbalanced. For example, BigVul contains many more non-vulnerable samples, which can make the model predict most samples as non-vulnerable.

To address this, we improve normalization, augment the datasets, and balance the resulting training data.

### BetterNormalization

The original vdgraph normalization is relatively coarse-grained. Some samples cannot be normalized as expected, and some transformed samples may introduce syntax errors. We therefore provide **BetterNormalization**, which is designed to produce more reliable normalized code before graph preprocessing and victim-model training.

```text
raw source sample
    -> BetterNormalization
    -> normalized source sample
    -> graph preprocessing
    -> victim-model training
```

### Dataset augmentation and balancing

We use CloneGen-style TXL code transformations for semantics-preserving data augmentation. Since the original TXL rules do not robustly support many modern C++ constructs, we adapt the transformations to better support **C++17** syntax. This allows most samples in the datasets to be transformed successfully while preserving the original label.

```text
original source sample
    -> C++17-compatible TXL transformation
    -> semantically equivalent variant
    -> augmented dataset
```

After augmentation, we undersample the majority class to obtain an approximately balanced vulnerable / non-vulnerable ratio:

```text
augmented dataset
    -> count vulnerable and non-vulnerable samples
    -> undersample the majority class
    -> balanced training set
```

This balancing step helps avoid majority-class collapse and produces stronger victim models for adversarial evaluation.

### Graph preprocessing and graph-to-source mapping

The base graph preprocessing pipeline follows vdgraph:

```text
source code
    -> Joern parsing
    -> graph representation
    -> Word2Vec training
    -> graph embedding
    -> GNN victim-model training
```

For details of graph embedding and Word2Vec training, refer to the original vdgraph implementation.

ExplAtk additionally needs graph-to-source mapping because graph explanations must be projected back to source statements and identifiers. We modify `joern_graph_gen` to export JSON files containing:

```text
dot graph information
source-code line numbers
graph node / source-line mapping information
```

These files are used by the explanation-to-source projection module to compute source-level importance scores `S(z)`.

---

## 3. Attack Methods

All attacks use the same high-level convention:

```text
<method>/src/attack/<attack_method_name>.py
```

Supported methods:

| Method | Type | Entry file |
|---|---|---|
| MHM | Baseline | `MHM/src/attack/mhm.py` |
| ALERT | Baseline | `Alert/src/attack/alert.py` |
| DIP | Baseline | `DIP/src/attack/dip.py` |
| CODA | Baseline | `CODA/src/attack/coda.py` |
| MOAA | Baseline | `MOAA/src/attack/moaa.py` |
| ExplAtk | Proposed method | `ExplAtk/src/attack/explatk.py` |

Some baseline implementations are not fully open-sourced, while others do not directly support GNN-based vulnerability detection, where attacks require source preprocessing, graph construction, graph embedding, and victim-model wrapper queries. Therefore, we re-implement all baselines by following the corresponding papers and adapting them to the same GNN-based evaluation setting.

### Run an attack from a method directory

For example:

```bash
cd ExplAtk
PYTHONPATH=. python src/attack/explatk.py
```

The same pattern applies to `MHM`, `Alert`, `DIP`, `CODA`, and `MOAA`.

### Optional unified attack wrapper

This repository does not require a separate unified runner script. However, for large-scale evaluation, users may write a lightweight wrapper that dispatches attacks by name and collects all outputs in the same format.

```python
import importlib
from typing import Any, Callable, Dict

from common.attack_result import AttackResult

AttackRunner = Callable[..., AttackResult]

def get_attack_runner(attack_name: str) -> AttackRunner:
    normalized = attack_name.strip().lower().replace("-", "_")
    mapping: Dict[str, tuple[str, str]] = {
        "mhm": ("MHM.src.attack.mhm", "run_mhm_attack"),
        "alert": ("Alert.src.attack.alert", "run_alert_attack"),
        "dip": ("DIP.src.attack.dip", "run_dip_attack"),
        "coda": ("CODA.src.attack.coda", "run_coda_attack"),
        "moaa": ("MOAA.src.attack.moaa", "run_moaa_attack"),
        "explatk": ("ExplAtk.src.attack.explatk", "run_expl_attack"),
    }
    module_name, function_name = mapping[normalized]
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def run_attack_by_name(
    attack_name: str,
    model_name: str,
    checkpoint_path: str,
    source_code: str,
    true_label: int,
    sample_id: str | int = "unknown",
    **attack_kwargs: Any,
) -> AttackResult:
    runner = get_attack_runner(attack_name)
    return runner(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        source_code=source_code,
        true_label=true_label,
        sample_id=sample_id,
        **attack_kwargs,
    )
```

---

## 4. Unified Attack Result

All attacks return the same `AttackResult` dataclass.

```python
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class AttackResult:
    sample_id: str | int
    attack_name: str
    model_name: str
    true_label: int
    original_pred: int
    original_true_conf: float
    is_attackable: bool
    success: bool
    query_count: int
    original_code: str
    final_variant: str
    best_variant_by_conf_drop: str
    first_success_variant: str | None
    final_pred: int
    final_true_conf: float
    best_true_conf: float
    success_true_conf: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

Key fields:

| Field | Meaning |
|---|---|
| `is_attackable` | Whether the original sample is suitable for attack, usually requiring `original_pred == true_label`. |
| `success` | Whether the attack flips the prediction. |
| `query_count` | Number of victim-model queries. |
| `final_variant` | Final generated code variant. |
| `best_variant_by_conf_drop` | Variant with the largest true-label confidence drop. |
| `first_success_variant` | First variant that successfully flips the prediction. |
| `best_true_conf` | Lowest true-label confidence observed during the attack. |

Recommended output layout:

```text
outputs/
    mhm_results.jsonl
    alert_results.jsonl
    dip_results.jsonl
    coda_results.jsonl
    moaa_results.jsonl
    explatk_results.jsonl
```

---

## 5. Project Structure

```text
.
├── Alert/
├── CODA/
├── DIP/
├── ExplAtk/
│   ├── explain/
│   │   ├── coca_explainer.py
│   │   ├── mapping.py
│   │   └── robust_explainer.py
│   └── src/
│       ├── attack/
│       ├── model/
│       └── utils/
├── MHM/
├── MOAA/
├── common/
│   ├── attack_result.py
│   ├── attack_result_writer.py
│   ├── config_loader.py
│   └── utils/
├── preprocess/
│   ├── joern-1.1.172/
│   ├── joern_graph_gen.py
│   └── CLONEGEN/
├── config.yaml
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 6. Notes

- Attacks are usually evaluated only on samples that are correctly classified by the victim model.
- If `is_attackable=False`, the sample is typically excluded from the attack success rate denominator.
- Identifier-renaming attacks should avoid C/C++ keywords, scope conflicts, syntax errors, and semantic changes.
- Structure-guided attacks should preserve the original program semantics.
