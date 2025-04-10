# CaseReportBench: Benchmarking LLMs for Dense Information Extraction in Clinical Case Reports

_Official repository for the **accepted** CHIL 2025 paper_  
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue.svg)](https://huggingface.co/datasets/cxyzhang/consolidated_expert_validated_denseExtractionDataset)
[![Conference](https://img.shields.io/badge/Accepted%20at-CHIL%202025-4b8bbe)](https://chil.ahli.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![Cite this](https://img.shields.io/badge/Cite-BibTeX-blue)](#-citation)

---

## 📘 Overview

**CaseReportBench** is the first benchmark designed for **dense information extraction** from clinical case reports, focused on **rare diseases**, especially **Inborn Errors of Metabolism (IEMs)**. This benchmark evaluates how well **large language models (LLMs)** can extract structured, clinically relevant data across **14 system-level categories**, such as Neurology, History, Lab/Imaging, and Musculoskeletal (MSK).

**Key Contributions:**
- A curated dataset of **138 expert-annotated case reports**.
- Dense extractions across 14 predefined diagnostic categories.
- Evaluation of LLMs including **Qwen2*, **Qwen2.5**, **LLaMA3**, and **GPT-4o**.
- Novel prompting strategies: **Filtered Category-Specific Prompting (FCSP)**, **Uniform Category-Specific Prompting (UCP)**, and **Unified Global Prompting (UGP)**.
- Expert clinical assessment of model outputs.

---

## 🧩 Source Code Overview (`src/`)

The `src/` folder contains all key components for dataset construction, prompting logic, and LLM evaluation:

| Folder | Description |
|--------|-------------|
| `dataset_construction/` | Scripts to process PMC-OA case reports, filter IEM cases, and structure data into prompt-ready JSON. Includes code for expert annotation merging and TSR filtering. |
| `prompting/` | Code to run LLMs (local or API) using structured prompts (UGP, FCSP, UCP), including category-specific and unified templates. Also supports prompt instantiation and model input formatting. |
| `benchmarking_llms/` | Evaluate LLM dense information extractions against gold expert-crafted annotations, and compute all metrics (TSR, EM, hallucination, etc). |

## 🧪 Setup Instructions

To set up the environment using Conda:
```bash
conda env create -f environment.yaml
conda activate CaseReportBench
```


## 📦 Dataset Access

The dataset is available on the Hugging Face Hub:  
👉 https://huggingface.co/datasets/cxyzhang/consolidated_expert_validated_denseExtractionDataset

To load it in Python:

```python
from datasets import load_dataset

dataset = load_dataset("cxyzhang/consolidated_expert_validated_denseExtractionDataset")

```

## 🔓 License

- **Code**: MIT License ([LICENSE.txt](https://github.com/cindyzhangxy/CaseReportBench/blob/main/LICENSE.txt))
- **Dataset**: CC BY-NC 4.0 ([DATA_LICENSE.txt](https://github.com/cindyzhangxy/CaseReportBench/blob/main/DATA_LICENSE.txt))


> The dataset is derived from the PubMed Central Open Access Subset and is for **non-commercial academic use** only.

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{casereportbench2025,
  title     = {CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports},
  author    = {Xiao Yu Cindy Zhang, Carlos R. Ferreira, Francis Rossignol, Raymond T. Ng, Wyeth Wasserman, Jian Zhu},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning (CHIL)},
  year      = {2025},
  note      = {Accepted}
}

