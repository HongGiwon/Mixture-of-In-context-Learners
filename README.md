
# Mixture-of-In-Context-Learners (MOICL)

This repository contains implementations for running **Mixture-of-In-Context-Learners (MOICL)** on both classification and generation tasks.\
The method partitions in-context demonstrations into multiple sets and learns **scalar weights** or a **hypernetwork** to combine them effectively.

## 1. Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Scripts Overview

The repository contains two main entry points:

| Script | Task Type      | Description                                                                                                            |
| ------ | -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| \`\`   | Classification | Runs MOICL on text classification datasets such as SST-2, RTE, hate/offensive speech detection, FEVER, PAWS, and QNLI. |
| \`\`   | Generation     | Runs MOICL on a math reasoning task (GSM8K).                                                                      |

---

## 3. Input Parameters

### \*\*Classification: \*\*\`\`

| Argument             | Type    | Default                                 | Description                                                                         |
| -------------------- | ------- | --------------------------------------- | ----------------------------------------------------------------------------------- |
| `--model_name`       | `str`   | `"meta-llama/Meta-Llama-3-8B-Instruct"` | Pretrained LLM name.                                                                |
| `--n_samples`        | `int`   | `1`                                     | Number of demonstrations per set.                                                   |
| `--n_sets`           | `int`   | `30`                                    | Number of demonstration sets.                                                       |
| `--seed`             | `int`   | `42`                                    | Random seed.                                                                        |
| `--dataset`          | `str`   | `None`                                  | Dataset name (`sst`, `rte`, `hate`, `offensive`, `fever`, `paws`, `qnli`).          |
| `--n_epoch`          | `int`   | `5`                                     | Training epochs.                                                                    |
| `--lr`               | `float` | `0.1`                                   | Learning rate for hypernet/scalar weights.                                          |
| `--accum_step`       | `int`   | `12`                                    | Gradient accumulation steps.                                                        |
| `--train_instance`   | `int`   | `-1`                                    | Limit number of training examples (`-1` = use all).                                 |
| `--noeval`           | flag    | `False`                                 | Disable evaluation during training.                                                 |
| `--use_cache`        | flag    | `False`                                 | Cache model outputs for speed.                                                      |
| `--hyper_model_name` | `str`   | `""`                                    | Hypernetwork model name (e.g., `"google-t5/t5-small"`). Empty = use scalar weights. |

---

### \*\*Generation: \*\*\`\`

| Argument             | Type    | Default                        | Description                             |
| -------------------- | ------- | ------------------------------ | --------------------------------------- |
| `--model_name`       | `str`   | `"meta-llama/Meta-Llama-3-8B"` | Pretrained LLM name.                    |
| `--hyper_model_name` | `str`   | `"google-t5/t5-small"`         | Hypernetwork model name.                |
| `--n_samples`        | `int`   | `1`                            | Demonstrations per set.                 |
| `--seed`             | `int`   | `31`                           | Random seed.                            |
| `--n_epoch`          | `int`   | `1`                            | Training epochs.                        |
| `--lr`               | `float` | `0.0001`                       | Learning rate.                          |
| `--n_sets`           | `int`   | `6`                            | Number of sets.                         |
| `--accum_step`       | `int`   | `12`                           | Gradient accumulation steps.            |
| `--scalar_weights`   | flag    | `False`                        | Use scalar weights instead of hypernet. |
| `--train_instance`   | `int`   | `-1`                           | Limit number of training examples.      |
| `--use_cache`        | flag    | `False`                        | Enable output caching.                  |

---

## 4. Datasets

The scripts expect datasets to be available via [Hugging Face Datasets](https://huggingface.co/datasets).

**Supported classification datasets:**

- `sst` – SST-2 sentiment classification
- `rte` – Recognizing Textual Entailment
- `hate` – Hate speech detection
- `offensive` – Offensive language classification
- `fever` – Fact verification
- `paws` – Paraphrase Adversaries from Word Scrambling
- `qnli` – Question-answer entailment

**Supported generation datasets:**

- `gsm8k` – Grade school math problems (via `openai/gsm8k`)

---

## 5. Example Usage

### **Classification example (TweetEval offensive with scalar weights):**

```bash
python run_moicl_classification.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset offensive \
  --n_samples 3 \
  --n_sets 10 \
  --n_epoch 3 \
  --lr 0.05 \
  --seed 42
```

### **Classification example (TweetEval offensive with hypernetwork):**

```bash
python run_moicl_classification.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --dataset offensive \
  --n_samples 3 \
  --n_sets 10 \
  --n_epoch 3 \
  --lr 0.05 \
  --seed 42 \
  --hyper_model_name google-t5/t5-small
```

---

### **Generation example (GSM8K with hypernetwork):**

```bash
python run_moicl_generation_gsm.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --n_samples 1 \
  --n_sets 6 \
  --n_epoch 1 \
  --lr 0.0001 \
  --seed 31 \
  --hyper_model_name google-t5/t5-small
```

### **Generation example (GSM8K with scalar weights):**

```bash
python run_moicl_generation_gsm.py \
  --model_name meta-llama/Meta-Llama-3-8B \
  --n_samples 1 \
  --n_sets 6 \
  --n_epoch 1 \
  --lr 0.0001 \
  --seed 31 \
  --scalar_weights
```

---

## 6. Notes

- All model names should be accessible from the Hugging Face Hub.
- Training large models (e.g., LLaMA 3 8B) requires sufficient GPU memory.

---

## 7. Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{hong-etal-2025-mixtures,
    title = "Mixtures of In-Context Learners",
    author = "Hong, Giwon  and
      Van Krieken, Emile  and
      Ponti, Edoardo  and
      Malkin, Nikolay  and
      Minervini, Pasquale",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1277/",
    doi = "10.18653/v1/2025.acl-long.1277",
    pages = "26332--26351",
    ISBN = "979-8-89176-251-0"
}
```