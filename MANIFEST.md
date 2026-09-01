# MANIFEST

## Software versions used for the reported runs

| Package | Version |
|---|---|
| Python | 3.11.15 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| matplotlib | 3.11.1 |
| tqdm | 4.69.0 |
| ase | 3.29.0 |
| spglib | 2.7.0 |
| pymatgen | 2026.5.4 |
| python-dotenv | 1.2.2 |
| torch | 2.5.1 (pinned for broad PyTorch Geometric wheel compatibility rather than the latest PyPI release, 2.13.0, whose PyG extension wheels were not available at time of writing) |
| torch-geometric | 2.6.1 |
| torch-scatter | 2.1.2 |
| torch-sparse | 0.6.18 |
| torch-cluster | 1.6.3 |
| torch-spline-conv | 1.2.2 |

## Operating system

macOS (Darwin, arm64), verified in the sandboxed analysis environment used to develop and validate `create_scalar/carve.py`. The Dockerfile targets Ubuntu 22.04 with CUDA 12.1 for GPU-backed runs of the GNN baselines and the carving/validation pipeline.

## LLM model run manifest

Source: ESI, Section "Model registry and API parameters", verbatim.

Access window: January-February 2026, via a unified inference API (OpenRouter). Sampling: temperature 0.0 (greedy decoding), provider-default maximum completion length, N=5 independent queries per sample for the consistency metric, prompts submitted independently with no shared conversation state.

| Model | Identifier |
|---|---|
| Claude 3 Haiku | `anthropic/claude-3-haiku` |
| Seed 1.6 Flash | `bytedance-seed/seed-1.6-flash` |
| DeepSeek v3.2 | `deepseek/deepseek-v3.2` |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` |
| LLaMA 4 Maverick | `meta-llama/llama-4-maverick` |
| Ministral 3 14B | `mistralai/ministral-14b-2512` |
| Nemotron 3 Nano 30B | `nvidia/nemotron-3-nano-30b-a3b` |
| GPT-5 Mini | `openai/gpt-5-mini` |
| o3-mini | `openai/o3-mini` |
| Grok 4.1 Fast | `x-ai/grok-4.1-fast` |
