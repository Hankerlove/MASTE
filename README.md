# MASTE: Multi-Agent Pipeline for Zero-Shot Aspect Sentiment Triplet Extraction

This repository contains the official implementation of **MASTE**, a
training-free multi-agent framework for zero-shot Aspect Sentiment Triplet
Extraction (ASTE).

Given a review sentence, ASTE aims to extract all
`(aspect, opinion, sentiment)` triples. MASTE decomposes this structured
prediction problem into four sequential LLM agents:

1. **Aspect Extraction Agent** identifies candidate aspect terms.
2. **Opinion Extraction Agent** finds opinion expressions conditioned on each
   extracted aspect.
3. **Sentiment Reasoning Agent** assigns polarity to each aspect-opinion pair.
4. **Consistency Check Agent** verifies and revises the predicted triplets.

## Repository Structure

```text
src/
  agents/
    aspect_agent.py       # Aspect extraction agent
    opinion_agent.py      # Opinion extraction agent
    sentiment_agent.py    # Sentiment reasoning agent
    consistency_agent.py  # Consistency checking agent
  baselines.py            # Zero-shot, few-shot, direct, and CoT baselines
  data_loader.py          # ASTE dataset loader
  evaluate.py             # Exact-match precision, recall, and F1
  llm_client.py           # OpenAI-compatible LLM client
  pipeline.py             # MASTE pipeline orchestration

experiments/
  run_main.py             # Main experiment entry point
  run_ablation.py         # Ablation experiment entry point

requirements.txt
```

Local test files, paper sources, datasets, generated results, logs, and
paper-writing helper scripts are intentionally excluded from the public release.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure an OpenAI-compatible API endpoint:

```bash
export OPENAI_API_KEY=sk-...

# Optional: use another OpenAI-compatible provider
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Data Preparation

The experiment scripts expect ASTE-Data-V2 files under:

```text
data/aste/SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/
```

The expected domain folders are:

```text
14res/
14lap/
15res/
16res/
```

Each folder should contain the corresponding `train_triplets.txt`,
`dev_triplets.txt`, and `test_triplets.txt` files. The dataset is not included in
this repository; please obtain it from the original ASTE benchmark release and
place it at the path above.

## Running Experiments

Run MASTE on a single domain:

```bash
PYTHONPATH=. python experiments/run_main.py \
  --method maste \
  --model gpt-4o \
  --domains 14res \
  --split test \
  --output_dir results/maste_gpt4o
```

Run all four ASTE domains:

```bash
PYTHONPATH=. python experiments/run_main.py \
  --method maste \
  --model gpt-4o \
  --domains 14res 14lap 15res 16res \
  --split test \
  --output_dir results/maste_gpt4o
```

Run a single-call zero-shot baseline:

```bash
PYTHONPATH=. python experiments/run_main.py \
  --method zero_shot \
  --model gpt-4o \
  --domains 14res 14lap 15res 16res \
  --split test \
  --output_dir results/zero_shot_gpt4o
```

Run a chain-of-thought baseline:

```bash
PYTHONPATH=. python experiments/run_main.py \
  --method cot \
  --model gpt-4o \
  --domains 14res 14lap 15res 16res \
  --split test \
  --output_dir results/cot_gpt4o
```

Run an ablation experiment:

```bash
PYTHONPATH=. python experiments/run_ablation.py \
  --model gpt-4o \
  --domain 14res \
  --split test \
  --output_dir results/ablation/gpt4o_14res
```

## Output Format

Each experiment writes one JSON file per domain and a `summary.json` file under
the specified output directory. Per-domain files contain the original sentence,
gold triplets, predicted triplets, and exact-match evaluation metrics.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{mastedraft,
  title = {MASTE: A Multi-Agent Pipeline for Zero-Shot Aspect Sentiment Triplet Extraction},
}
```
