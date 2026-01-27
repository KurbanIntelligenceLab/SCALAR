# SCALAR: Quantifying Structural Hallucination, Consistency, and Reasoning Gaps in Materials Foundation Models

Large language models and foundation models are increasingly applied to scientific reasoning tasks in materials science, yet their behavior under physically structured distribution shifts remains poorly understood. We introduce **SCALAR** (**S**tructural **C**onsistency **A**nd **L**ogic **A**cross **R**egimes), a task-based benchmark for evaluating geometric scale generalization and its connection to hallucination, consistency, and reasoning in materials foundation models.

Given a canonical crystal representation, models are required to reason about derived nanoparticle structures obtained through deterministic supercell expansion and geometric truncation across a wide range of length scales, spanning systems from a few atoms to more than eighteen thousand atoms and totaling over 100,000 structures derived from DFT-validated unit cells. SCALAR defines three controlled tasks: direct CIF→property prediction, a Chain-of-Thought (CoT) variant that differs only by the inclusion of explicit physics-grounded intermediate reasoning, and an inverse retrieval task that identifies the correct crystal from candidate sets given target properties. Model outputs are evaluated using structured metrics capturing numeric error, constraint-based hallucination, cross-prompt consistency, monotonic reasoning across radii, output validity, and physical distance and regret in inverse retrieval. To isolate the effect of explicit reasoning, we report percent changes between CoT and non-CoT prompting in 1-shot and 3-shot regimes, aggregated over in- and out-of-distribution splits. Experiments across diverse foundation models reveal large, model-dependent shifts under explicit reasoning—often reducing hallucination and numeric error, but frequently destabilizing consistency or output validity—demonstrating that geometric scale generalization and scientific reliability cannot be inferred from accuracy alone.

---

## Creating the SCALAR dataset

The `create_scalar` pipeline builds the scalar dataset from raw data (directory or zip). It extracts CIFs into `unit_cells/`, extracts XYZ files into a temporary `materials/` layout, then runs quaternion-based rotation generation (using `create_scalar.config` and `ScalarQuaternionGenerator`) to produce rotated structures in `quaternions/`.

### Input: `scalar_raw`

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
