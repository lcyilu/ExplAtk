# ExplAtk: Explanation-Guided Adversarial Attacks on GNN-Based Vulnerability Detectors

ExplAtk studies adversarial attacks against graph neural network (GNN) based vulnerability detection models. The main idea is to use a graph explainer to identify prediction-critical graph nodes, graph edges, source statements, and identifiers, then use the explanation score `S(z)` to guide candidate generation and iterative adversarial search.

The repository includes several baselines, including **MHM**, **ALERT**, **DIP**, **CODA**, and **MOAA**, as well as the proposed method **ExplAtk**. Each attack method is implemented under the corresponding method directory, and the attack entry file follows the pattern:

```text
<method>/src/attack/<attack_method_name>.py
```

All attack methods return a unified `AttackResult` object to make evaluation and comparison consistent.

---

## 1. Environment Setup

This section describes how to reproduce the Python environment.

### 1.1 Create the Conda environment

From the project root directory, run:

```bash
conda env create -f environment.yml
conda activate explatk
```

If the environment name in `environment.yml` is different, use the name specified by the `name:` field.

### 1.2 Install additional Python dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Verify PyTorch and CUDA

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CUDA is unavailable, the code can still run on CPU, but victim model training, explanation generation, and attack search will be slower.

### 1.4 Check Joern dependencies

The project relies on Joern to parse C/C++ source code and export graph representations such as CPG and PDG. The expected Joern-related files are located under:

```text
preprocess/
preprocess/joern-1.1.172/
preprocess/joern_graph_gen.py
my-languages.so
```

Make sure the Joern executables are available:

```bash
chmod +x preprocess/joern-1.1.172/joern
chmod +x preprocess/joern-1.1.172/joern-parse
chmod +x preprocess/joern-1.1.172/joern-export
```

### 1.5 Configure local paths

Before preprocessing, training, or running attacks, check `config.yaml` and make sure the paths match your local machine.

Typical fields include:

```yaml
joern_path: preprocess/joern-1.1.172
dataset_path: <path_to_dataset>
processed_data_path: <path_to_processed_graphs>
checkpoint_path: <path_to_victim_model_checkpoint>
word2vec_path: <path_to_word_vectors>
codebert_path: <path_to_codebert>
codet5_path: <path_to_codet5>
output_dir: <path_to_outputs>
```

The exact field names should follow the actual `config.yaml` file in this repository.

---

## 2. Victim Model Training and Preprocessing

This section describes how the victim GNN-based vulnerability detectors are trained and how the datasets are prepared before training. Our victim-model setup is built on the prior work **Interpreters for GNN-based Vulnerability Detection: Are We There Yet?** by Hu et al. and its released implementation, **vdgraph**. That work provides the original datasets, victim model implementations, and the preprocessing pipeline from source code to graph representation, Word2Vec training, graph embedding, and model training.

Reference:

```bibtex
@inproceedings{hu2023interpreters,
  title={Interpreters for GNN-based vulnerability detection: Are we there yet?},
  author={Hu, Yutao and Wang, Suyuan and Li, Wenke and Peng, Junru and Wu, Yueming and Zou, Deqing and Jin, Hai},
  booktitle={Proceedings of the 32nd ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages={1407--1419},
  year={2023}
}
```

The original repository is available at:

```text
https://github.com/CGCL-codes/vdgraph
```

### 2.1 Victim models

We use four representative GNN-based vulnerability detectors as victim models:

| Victim model | Description |
|---|---|
| Devign | A GNN-based vulnerability detector that learns from graph representations of source code. |
| IVDetect | A GNN-based vulnerability detector that uses program-dependence information and attention-based graph convolution. |
| Reveal | A GNN-based vulnerability detector designed for learning vulnerability-related graph representations. |
| DeepWukong | A graph-based vulnerability detector that normalizes source code and learns from vulnerability-oriented program slices. |

Each victim detector is trained on each dataset, producing 16 victim settings in total:

```text
4 victim models x 4 datasets = 16 victim settings
```

The attack code queries these victim models through the corresponding `ModelWrapper`, which provides a unified interface for obtaining:

```text
prediction
true-label confidence
margin
```

### 2.2 Training datasets

The victim models are trained and evaluated on four vulnerability datasets:

| Dataset | Source / construction | Train / Val / Test split |
|---|---|---|
| Devign | Augmented from the original Devign dataset | 58,462 / 19,490 / 19,490 |
| BigVul | Augmented from the original BigVul dataset | 45,511 / 15,209 / 15,209 |
| CWE119 | Derived from the augmented BigVul dataset by selecting CWE-119 vulnerability samples | 15,492 / 5,167 / 5,167 |
| Reveal | Augmented from the original Reveal dataset | 14,288 / 4,764 / 4,764 |

The original Devign, BigVul, and Reveal datasets are inherited from the vdgraph workflow. CWE119 is derived from BigVul by selecting samples associated with CWE-119. The dataset introduction and the model implementations follow the prior work, while this repository adds additional preprocessing, augmentation, and balancing steps to obtain stronger and more reliable victim models.

### 2.3 Motivation for additional preprocessing

When training the four victim models directly on the original BigVul, Devign, and Reveal datasets, we observed two practical issues.

First, the base performance of the trained victim models was often unsatisfactory. A weak victim model makes adversarial evaluation less meaningful, because attack success may be caused by poor training rather than by a real robustness weakness.

Second, some datasets are highly imbalanced. BigVul, for example, contains far more non-vulnerable samples than vulnerable samples. Under this imbalance, the victim model may learn a degenerate decision rule that predicts most or all samples as non-vulnerable, making it difficult to train an effective vulnerability detector.

For these reasons, we augment the training data and then balance the augmented datasets before victim model training.

### 2.4 BetterNormalization

The original vdgraph workflow provides a normalization procedure. However, we found that the original normalization is relatively coarse-grained. For many samples, it does not always perform the intended normalization reliably, and in some cases the transformed code may even contain syntax errors.

To address this issue, this repository provides an improved normalization component, referred to as **BetterNormalization**. BetterNormalization is designed to preserve the intended normalization behavior while reducing invalid transformations and syntax-breaking cases.

The intended role of BetterNormalization is:

```text
raw source sample
    -> BetterNormalization
    -> normalized source sample
    -> graph preprocessing and victim training
```

This improves the quality of the normalized training samples and helps train more stable victim models.

### 2.5 Dataset augmentation with improved TXL transformations

To train stronger victim models, we perform semantics-preserving data augmentation before training. The augmentation is based on the TXL code transformation tools provided by CloneGen, with additional improvements in this repository.

The main improvement is compatibility with modern C++ syntax, especially C++17. The original CloneGen TXL transformations do not robustly cover many C++17 constructs that appear in real vulnerability datasets. We extend and adapt the TXL transformation rules so that the majority of samples in the datasets can be transformed successfully.

The augmentation pipeline is:

```text
original source sample
    -> C++17-compatible TXL transformation
    -> semantically equivalent transformed variant
    -> augmented training dataset
```

The generated variants preserve the vulnerability label while increasing the diversity of the training data.

### 2.6 Dataset balancing by undersampling

After augmentation, we balance the datasets by undersampling the majority class. The goal is to avoid training victim models that are biased toward the non-vulnerable label.

The balancing procedure is:

```text
augmented dataset
    -> count vulnerable and non-vulnerable samples
    -> undersample the majority class
    -> construct a balanced training set
```

In our setting, the final augmented datasets are balanced with an approximately 1:1 ratio between vulnerable and non-vulnerable samples. This helps the victim models learn more meaningful decision boundaries and provides stronger targets for adversarial robustness evaluation.

### 2.7 Graph preprocessing and graph-to-source mapping

The preprocessing code is mainly located in:

```text
preprocess/
common/utils/gen_embedding.py
```

The base graph preprocessing pipeline follows vdgraph:

```text
source code
    -> Joern parsing
    -> graph representation
    -> Word2Vec training
    -> graph embedding
    -> GNN victim model training
```

For details such as graph embedding and Word2Vec training, please refer to the original vdgraph implementation.

In addition, ExplAtk requires graph-to-source mapping because the graph explanation results must be projected back to source statements and identifiers. Therefore, we modify the `joern_graph_gen` script so that it additionally exports JSON files containing:

```text
dot graph information
source-code line numbers
graph node / source-line mapping information
```

These mapping files are used by the explanation-to-source projection module to convert graph-level node and edge importance into source-level scores `S(z)`.

### 2.8 Joern version for reproducibility

Joern is used to parse source code and generate graph representations. Different Joern versions may produce slightly different graph structures, node labels, or line-number mappings. To make reproduction as close as possible to our experiments, this repository additionally provides the Joern version used in our experiments.

The official Joern repository is:

```text
https://github.com/joernio/joern
```

The bundled Joern directory is expected to be located under:

```text
preprocess/joern-1.1.172/
```

Make sure the Joern executables are available before preprocessing:

```bash
chmod +x preprocess/joern-1.1.172/joern
chmod +x preprocess/joern-1.1.172/joern-parse
chmod +x preprocess/joern-1.1.172/joern-export
```

### 2.9 Training workflow

A recommended victim-model training workflow is:

```bash
# Step 1: normalize and augment the datasets
# Use BetterNormalization and the improved C++17-compatible TXL transformations.

# Step 2: balance the augmented datasets
# Undersample the majority class to obtain an approximately 1:1 label ratio.

# Step 3: generate graph data and graph-to-source mapping files
python preprocess/joern_graph_gen.py

# Step 4: train a victim GNN
python ExplAtk/src/model/model.py
```

If your local repository uses a different training script, use that script while keeping the same preprocessing outputs and checkpoint format expected by the attack wrappers.
## 3. Attack Methods

This repository provides a unified interface for all attack methods. Instead of using different result formats or evaluation scripts for different baselines, every attack method follows the same entry convention and returns the same `AttackResult` dataclass.

The supported attacks are:

```text
mhm
alert
dip
coda
moaa
explatk
```

Their method-specific implementations are located under the corresponding method directory:

```text
MHM/src/attack/mhm.py
Alert/src/attack/alert.py
DIP/src/attack/dip.py
CODA/src/attack/coda.py
MOAA/src/attack/moaa.py
ExplAtk/src/attack/explatk.py
```

### 3.1 Why we re-implement the baselines

Some baseline methods are not fully open-sourced, while others were originally designed for different model families or different code representations. In particular, attacks on GNN-based vulnerability detectors require additional steps such as source-code preprocessing, graph construction, graph embedding, model-wrapper queries, and sometimes graph-to-source alignment. These steps are not always supported by the original implementations of prior attacks.

Therefore, we implement all baselines in this repository by following the corresponding papers and adapting them to the same GNN-based vulnerability-detection setting. This gives all methods the same input format, victim-model query interface, and output format, making the comparison more controlled and reproducible.

### 3.2 Running attacks from each method directory

One way to run an attack is to enter the corresponding method directory and execute the attack script directly.

For example:

```bash
cd ExplAtk
PYTHONPATH=. python src/attack/explatk.py
```

The same convention applies to `MHM`, `ALERT`, `DIP`, `CODA`, and `MOAA`.

### 3.3 Optional unified attack wrapper

This repository does not require a separate unified runner script. Each attack can be executed from its own method directory, as described above. However, for large-scale evaluation, it is often convenient to write a lightweight wrapper that dispatches different attacks by name and collects their outputs in the same format.

The following code shows the core idea. Reproduction users may adapt it to their own experiment scripts, configuration files, checkpoint paths, and dataset loaders.

```python
import importlib
from typing import Any, Callable, Dict

from common.attack_result import AttackResult

AttackRunner = Callable[..., AttackResult]


def get_attack_runner(attack_name: str) -> AttackRunner:
    """Load the attack function by attack name."""
    normalized = attack_name.strip().lower().replace("-", "_")

    mapping: Dict[str, tuple[str, str]] = {
        "mhm": ("MHM.src.attack.mhm", "run_mhm_attack"),
        "alert": ("Alert.src.attack.alert", "run_alert_attack"),
        "dip": ("DIP.src.attack.dip", "run_dip_attack"),
        "coda": ("CODA.src.attack.coda", "run_coda_attack"),
        "moaa": ("MOAA.src.attack.moaa", "run_moaa_attack"),
        "explatk": ("ExplAtk.src.attack.explatk", "run_expl_attack"),
    }

    if normalized not in mapping:
        supported = ", ".join(sorted(mapping.keys()))
        raise ValueError(
            f"Unsupported attack_name '{attack_name}'. Supported: {supported}"
        )

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
    """Run one attack sample through a unified interface."""
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


## 4. Unified Attack Result Format

All attacks return a unified `AttackResult` dataclass. This makes it easier to compare baselines and the proposed method under the same evaluation protocol.

The result definition is:

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

Field descriptions:

| Field | Description |
|---|---|
| `sample_id` | Sample identifier |
| `attack_name` | Attack method name |
| `model_name` | Victim model name |
| `true_label` | Ground-truth label |
| `original_pred` | Victim model prediction on the original sample |
| `original_true_conf` | Confidence of the true label on the original sample |
| `is_attackable` | Whether the sample is attackable, usually requiring `original_pred == true_label` |
| `success` | Whether the attack succeeds |
| `query_count` | Number of victim model queries |
| `original_code` | Original source code |
| `final_variant` | Final adversarial variant generated by the attack |
| `best_variant_by_conf_drop` | Variant with the largest true-label confidence drop |
| `first_success_variant` | First variant that flips the prediction |
| `final_pred` | Prediction on the final variant |
| `final_true_conf` | True-label confidence on the final variant |
| `best_true_conf` | Lowest true-label confidence observed during the attack |
| `success_true_conf` | True-label confidence when the first successful variant is found |

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

The core repository structure is:

```text
.
├── Alert/
│   └── src/
│       ├── attack/
│       ├── model/
│       └── utils/
├── CODA/
│   └── src/
│       ├── attack/
│       ├── model/
│       └── utils/
├── DIP/
│   └── src/
│       ├── attack/
│       ├── model/
│       └── utils/
├── ExplAtk/
│   ├── explain/
│   │   ├── coca_explainer.py
│   │   ├── mapping.py
│   │   └── robust_explainer.py
│   └── src/
│       ├── attack/
│       │   ├── explatk.py
│       │   └── ts_transforms.py
│       ├── model/
│       └── utils/
├── MHM/
│   └── src/
│       ├── attack/
│       ├── model/
│       └── utils/
├── MOAA/
│   └── src/
│       ├── attack/
│       ├── model/
│       └── utils/
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

1. Attacks are usually performed only on samples that are correctly classified by the victim model:

   ```text
   original_pred == true_label
   ```

2. If `is_attackable` is `False`, the sample is usually excluded from the attack success rate denominator.

3. For identifier-renaming attacks, generated candidates should satisfy the following constraints:

   - The candidate is not a C/C++ keyword.
   - The candidate does not conflict with identifiers in the current scope.
   - The candidate preserves program semantics.
   - The candidate does not remove the original vulnerability.

4. For structure-guided attacks, transformations should preserve the original program semantics.

5. ExplAtk uses graph explanations as attack guidance signals. These explanations should not be interpreted as formal proof of the real vulnerability root cause.
