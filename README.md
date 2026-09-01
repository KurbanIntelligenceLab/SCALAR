# SCALAR: Quantifying Structural Hallucination, Consistency, and Reasoning Gaps in Materials Foundation Models

Large language models and foundation models are increasingly applied to scientific reasoning tasks in materials science, yet their behavior under physically structured distribution shifts remains poorly understood. We introduce **SCALAR** (**S**tructural **C**onsistency **A**nd **L**ogic **A**cross **R**egimes), a task-based benchmark for evaluating geometric scale generalization and its connection to hallucination, consistency, and reasoning in materials foundation models.

Given a canonical crystal representation, models are required to reason about derived nanoparticle structures obtained through deterministic supercell expansion and geometric truncation across a wide range of length scales, spanning systems from a few atoms to more than eighteen thousand atoms and totaling over 100,000 structures derived from DFT-validated unit cells. SCALAR defines three controlled tasks: direct CIF→property prediction, a Chain-of-Thought (CoT) variant that differs only by the inclusion of explicit physics-grounded intermediate reasoning, and an inverse retrieval task that identifies the correct crystal from candidate sets given target properties. Model outputs are evaluated using structured metrics capturing numeric error, constraint-based hallucination, cross-prompt consistency, monotonic reasoning across radii, output validity, and physical distance and regret in inverse retrieval. To isolate the effect of explicit reasoning, we report percent changes between CoT and non-CoT prompting in 1-shot and 3-shot regimes, aggregated over in- and out-of-distribution splits. Experiments across diverse foundation models reveal large, model-dependent shifts under explicit reasoning—often reducing hallucination and numeric error, but frequently destabilizing consistency or output validity—demonstrating that geometric scale generalization and scientific reliability cannot be inferred from accuracy alone.

---

## Resource expectations

**Hardware.** Phase-I carving (`create_scalar/carve.py`) and dataset assembly run on CPU; the largest structures reach approximately 18,000 atoms and complete on a single core in well under a minute. GNN baseline training (`experiments/generating_from_cif/gnn_baselines.py`) uses PyTorch Geometric and benefits from a CUDA GPU; the Dockerfile targets CUDA 12.1. LLM benchmark tasks issue network requests only and require no local GPU or significant CPU.

**Runtime.** Full Phase-I carving and validation across all 83 materials and 21 radii (1,743 structures) completes in a few minutes on 8 parallel worker processes on a standard laptop-class CPU. Quaternion rotation generation to the full ~100,000-structure benchmark is the dominant dataset-build cost and is I/O bound by file writes rather than CPU. LLM benchmark runs are dominated by API round-trip latency (seconds per call) rather than local compute.

**Disk footprint.** The deposited 1,743 base XYZ structures (`scalar_raw/`) total 276.5 MB (mean 158.6 KB per file, measured directly from the deposited files). The full benchmark augments these base structures by rigid rotation to approximately 100,000 total structures, an augmentation factor of 100,000 / 1,743 ≈ 57.4x. Scaling the measured base size by this factor gives a projected disk footprint of approximately **14.8 GiB (15.9 GB)** for the full rotated dataset.

**API cost.** The seven LLM benchmark scripts (`cif_to_properties_zeroshot`, `cif_to_properties_1shot`, `cif_to_properties_3shot`, `reasoning_gap_1shot`, `reasoning_gap_3shot`, `inverse_with_3_possible`, `inverse_with_5_possible`) each issue one API call per (material, model, repeat) combination, with one prompt per material covering all target radii. At the default full-dataset configuration (83 materials, the 10 models in `experiments/benchmark_models.py`, and the N=5 repeats per sample reported in the ESI for the consistency metric), this is 83 x 10 x 5 x 7 = **29,050 API calls**. This count is derived from the script structure and the N=5 repeat count stated in the ESI, not from execution logs, which the repository does not retain; the scripts' own `--repeats` CLI defaults are 3 for the property-prediction and chain-of-thought scripts and 1 for the inverse-retrieval scripts, so a run at those defaults issues correspondingly fewer calls. Using a rough per-call estimate of approximately 1,200 input tokens (CIF text plus prompt template) and approximately 500 output tokens (structured JSON response), this totals approximately 34.9 million input tokens and 14.5 million output tokens across the full benchmark. The dollar cost of these tokens depends entirely on the provider and model pricing in effect at the time of the run (OpenRouter passes through per-model provider pricing that varies by more than 100x between the smallest and largest models in the registry) and is not stated here as a fixed figure.

---

## Creating the SCALAR dataset

The `create_scalar` pipeline builds the scalar dataset from raw data (directory or zip). It extracts CIFs into `unit_cells/`, extracts XYZ files into a temporary `materials/` layout, then runs quaternion-based rotation generation (using `create_scalar.config` and `ScalarQuaternionGenerator`) to produce rotated structures in `quaternions/`.

### Input: `scalar_raw`

The seed inputs are archived on Zenodo, DOI [10.5281/zenodo.20631920](https://doi.org/10.5281/zenodo.20631920), which always resolves to the latest version. Download `scalar_raw.zip` from there into the repository root before running the pipeline:

```bash
curl -L -o scalar_raw.zip "https://zenodo.org/records/22234497/files/scalar_raw.zip?download=1"
```

- **Directory** `scalar_raw/` or **zip** `scalar_raw.zip`.
- **CIFs:** in `scalar_raw/cifs/` or at the root of `scalar_raw/`.
- **XYZ files:** in `scalar_raw/materials/` or at the root of `scalar_raw/`.  
  Filenames must match `{Material}_R{6..30}.xyz` (e.g. `MAPbI3_R12.xyz`).

### Output: `scalar/`

```
scalar/
├── unit_cells/          # CIF files
│   └── {Material}.cif
└── quaternions/         # Rotated XYZ structures
    └── {Material}/
        └── R{x}/
            └── xyz/
                └── rot_*.xyz
```

### How to run

From the **project root** (where `create_scalar/` lives):

```bash
# Default: input = scalar_raw (dir or scalar_raw.zip), output = scalar/
python -m create_scalar.create_scalar

# Custom paths
python -m create_scalar.create_scalar --raw-data scalar_raw --output scalar
python -m create_scalar.create_scalar --raw-data scalar_raw.zip --output scalar

# Short flags
python -m create_scalar.create_scalar -r scalar_raw -o scalar

# Phase-I: carve base XYZ structures directly from the deposited CIFs instead of
# copying the deposited XYZ files (supercell construction, carving-centre
# selection, spherical truncation at R=10..30; see create_scalar/carve.py)
python -m create_scalar.create_scalar --raw-data scalar_raw --output scalar --from-cif
```

If `scalar_raw` does not exist, the script looks for `scalar_raw.zip` and uses it. Zip input is extracted to a temporary directory and cleaned up after the run.

### Configuration

Quaternion generation uses `create_scalar.config` (aligned with `utils/generate_quaternions`): `TARGET_TOTAL_FILES`, `SPLIT_FRACTIONS`, `MAX_ROTS_PER_FILE`, `ANGLE_BY_SPLIT`, radii splits (ID/OOD), etc. Adjust these in `create_scalar/config.py` if needed.

---

## Experiments

The benchmark implements the three SCALAR tasks. All experiments read from `scalar/unit_cells` and `scalar/quaternions` by default. **Run all commands from the project root.**

**Setup**

- Create the scalar dataset first (see [Creating the SCALAR dataset](#creating-the-scalar-dataset)).
- Install dependencies: `pip install -r requirements.txt`. The experiments use `python-dotenv` for `.env`; OpenRouter calls use the standard library.
- Model runs use [OpenRouter](https://openrouter.ai). Set `OPENROUTER_API_KEY` in the environment or in a `.env` file (loaded via `python-dotenv`).

### Task 1: CIF → property prediction

Direct prediction of nanoparticle properties (e.g. `num_atoms`, `composition`, `min_nn_distance`, `density`) from a crystal CIF at target radii R. Gold labels come from `rot_0.xyz`.

| Script | Description |
|--------|-------------|
| `experiments.generating_from_cif.cif_to_properties_zeroshot` | 0-shot: build prompts/gold, run OpenRouter, score |
| `experiments.generating_from_cif.cif_to_properties_1shot` | 1-shot: train-R examples in prompt |
| `experiments.generating_from_cif.cif_to_properties_3shot` | 3-shot: train-R examples in prompt |

Each has subcommands **`build`** (prompts + gold JSONL) and **`run`** (OpenRouter). Zeroshot also has **`score`** (compare predictions to gold).

**Zeroshot**

```bash
# Build prompts and gold (default: results/task_1_llm/zeroshot/)
python -m experiments.generating_from_cif.cif_to_properties_zeroshot build

# Run all registry models (or --models "model/id1,model/id2")
python -m experiments.generating_from_cif.cif_to_properties_zeroshot run

# Score predictions vs gold
python -m experiments.generating_from_cif.cif_to_properties_zeroshot score \
  --gold results/task_1_llm/zeroshot/gold.jsonl \
  --predictions results/task_1_llm/zeroshot/predictions_<model>.jsonl \
  --out results/task_1_llm/score.json
```

**1-shot / 3-shot**

```bash
# Build
python -m experiments.generating_from_cif.cif_to_properties_1shot build
python -m experiments.generating_from_cif.cif_to_properties_3shot build

# Run (uses same OpenRouter registry by default)
python -m experiments.generating_from_cif.cif_to_properties_1shot run
python -m experiments.generating_from_cif.cif_to_properties_3shot run
```

Use `--prompts-out`, `--gold-out`, `--prompts`, `--out-dir`, `--seed`, `--repeats`, `--max-prompts`, etc. as needed. Run `python -m experiments.generating_from_cif.cif_to_properties_zeroshot --help` (and likewise for 1shot/3shot) for full options.

### Task 2: Chain-of-Thought (CoT)

Same CIF→property setup, but with **explicit physics-based reasoning** (unit-cell volume, stoichiometry, nanoparticle scaling, nearest-neighbor intuition) before the final JSON. Used to study reasoning gaps and percent changes vs non-CoT.

| Script | Description |
|--------|-------------|
| `experiments.cot.reasoning_gap_1shot` | CoT 1-shot |
| `experiments.cot.reasoning_gap_3shot` | CoT 3-shot |

**Commands**

```bash
# Build prompts and gold
python -m experiments.cot.reasoning_gap_1shot build
python -m experiments.cot.reasoning_gap_3shot build

# Run on OpenRouter
python -m experiments.cot.reasoning_gap_1shot run
python -m experiments.cot.reasoning_gap_3shot run
```

Defaults: `results/task_1_cot_llm/1shot/` and `results/task_1_cot_llm/3shot/`. Use `--prompts-out`, `--gold-out`, `--prompts`, `--out-dir`, etc. to override.

### Task 3: Inverse retrieval

Given **target properties** (mass, density, mean NN distance), the model must choose the **correct CIF** among **3 or 5 candidates**.

| Script | Description |
|--------|-------------|
| `experiments.inverse.inverse_with_3_possible` | 3 candidates |
| `experiments.inverse.inverse_with_5_possible` | 5 candidates |

**Commands**

```bash
# Build (--cif-root, --xyz-root default to scalar/unit_cells, scalar/quaternions)
python -m experiments.inverse.inverse_with_3_possible build
python -m experiments.inverse.inverse_with_5_possible build

# Run
python -m experiments.inverse.inverse_with_3_possible run
python -m experiments.inverse.inverse_with_5_possible run
```

Outputs: `results/task_inverse_llm/3cand/` and `results/task_inverse_llm/5cand/`. Optional: `--r-values`, `--max-materials`, `--seed`, `--repeats`.

### Model registry

OpenRouter model IDs used by the benchmark are defined in **`experiments/benchmark_models.py`** (`OpenRouterModelRegistry.MODELS`). The `run` subcommands use this registry when `--models` is not provided. Override with `--models "model/id1,model/id2"`.

---

## Project layout

- **`create_scalar/`** — Dataset creation: config, raw→scalar pipeline, quaternion generation.
- **`experiments/`** — Benchmark scripts:
  - **`generating_from_cif/`** — Task 1 (CIF→property): zeroshot, 1-shot, 3-shot.
  - **`cot/`** — Task 2 (Chain-of-Thought): 1-shot, 3-shot.
  - **`inverse/`** — Task 3 (inverse retrieval): 3- and 5-candidate.
  - **`benchmark_models.py`** — OpenRouter model registry.

## License

See [LICENSE](LICENSE).
