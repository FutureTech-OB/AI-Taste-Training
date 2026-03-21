# AI-Taste-Training

This repository contains a reusable training and validation framework for research-article classification and prompt-based evaluation. The framework supports both API-based fine-tuning and local DeepSpeed training, with a shared data transformation layer and validation pipeline.

## Open Resources

- Hugging Face collection for open model weights and training data:
  `https://huggingface.co/collections/K1mG0ng/ai-taste-scientific`
- OpenAI fine-tuned models used in this project are access-only API models, not open weights.
- Because the data files are too large for direct repository distribution, reproducibility should start by downloading the corresponding JSONL files from the Hugging Face collection.
- Public data release is centered on JSONL. MongoDB support remains available for internal or local deployment workflows.

## Scope

The open-source-ready part of the codebase is primarily:

- `src/core`
- `src/practices/article`
- `scripts/sft`
- `scripts/validation`

The framework is built around four layers:

1. `src/core`
   Provides generic infrastructure for config loading, dataloaders, inference, validation metrics, and SFT backends.
2. `src/practices/article`
   Implements the article-specific prompt registry, data transformation, filtering, validation CLI, SFT CLI, and document models.
3. `scripts`
   Contains runnable shell entrypoints for validation and training workflows.

## Main Entry Points

### Validation

Run validation through:

```bash
python -m src.practices.article.validation \
  --model <model_name> \
  --provider <provider_name> \
  --prompt <prompt_name> \
  --entry rq_with_context \
  --data_source <mongodb|jsonl> \
  --db_name <db_name> \
  --data_file /path/to/validate.jsonl \
  --split validate \
  --subjects ECONOMICS SOCIOLOGY
```

Use `jsonl` for public validation pipelines and `mongodb` for internal or local database-backed workflows.

Useful shell wrappers:

- `scripts/validation/val.sh`
- `scripts/validation/val_lp.sh`
- `scripts/validation/val_ckpt.sh`

### SFT

Run SFT through:

```bash
python -m src.practices.article.sft \
  --trainer deepspeed \
  --data_source <mongodb|jsonl> \
  --db_name <db_name> \
  --data_file /path/to/train.jsonl \
  --split train \
  --subjects ECONOMICS SOCIOLOGY \
  --prompt social_science_rqcontext \
  --entry rq_with_context \
  --target_field rank \
  --model_path /path/to/base-model \
  --output_dir ./finetune/run_name
```

Use `jsonl` for public or portable training data pipelines, and `mongodb` for internal or local database-backed workflows.

Useful shell wrappers:

- `scripts/sft/article_sft_30B.sh`
- `scripts/sft/article_sft_4B.sh`
- `scripts/sft/openai_sft.sh`

## OpenAI Model List

The repository has been used with the following OpenAI fine-tuned model IDs for article-ranking validation:

| Scope | Model ID |
| --- | --- |
| OB | `ft:gpt-4.1-nano-2025-04-14:personal:ob-ob-rqcontext:DHKeHMNB` |
| OB | `ft:gpt-4.1-2025-04-14:personal:ob-ob-rqcontext:DHnLrzmY` |
| ECONOMICS | `ft:gpt-4.1-nano-2025-04-14:personal:social-science-rqc:DJWAxfSb` |
| Pooled | `ft:gpt-4.1-nano-2025-04-14:personal:eco-ob-social-scie:DJuAjWUp` |

These IDs are examples of evaluated fine-tuned models, not required defaults. Replace them with your own model IDs in validation scripts or local config when reproducing runs.

## Data Flow

The typical flow is:

1. Load records from MongoDB or JSONL.
2. Filter by `split`, `subject`, `type`, and optional year constraints.
3. Convert each article into a prompt/message format.
4. Train a model or run inference.
5. Store predictions in `val_outcome` or write transformed training data to downstream outputs.

Important article fields:

- `title`
- `journal`
- `published_year`
- `rank`
- `split`
- `subject`
- `entries`
- `val_outcome`

## Configuration

The runtime code reads these local config files:

- `assets/model.toml`
- `assets/database.toml`

The repository now keeps publishable templates in:

- `assets_example/model.toml.example`
- `assets_example/database.toml.example`
- `assets_example/ds_config_zero2.json`
- `assets_example/ds_config_zero3.json`

This layout is intentional:

- `.gitignore` currently ignores `/assets`
- `.gitignore` also ignores `*.toml`
- therefore the public templates must not live inside `assets/` and must not end with plain `.toml`

Recommended local setup for users:

1. Copy `assets_example/model.toml.example` to `assets/model.toml`
2. Copy `assets_example/database.toml.example` to `assets/database.toml`
3. Copy one of the DeepSpeed templates from `assets_example/` if you want to customize local training config
4. Fill in local credentials and private paths only in those ignored local files

Environment variables are still supported by some Python utilities and tests, but the shell training/validation entrypoints now read database settings from `assets/database.toml`.

## Recommended Minimal Public Layout

This public repository keeps:

- `src/core`
- `src/practices/article`
- `scripts/sft`
- `scripts/validation`
- `scripts/common`
- `scripts/merge`
- `assets_example/*.toml.example`
- `assets_example/ds_config_zero2.json`
- `assets_example/ds_config_zero3.json`

Users should provide their own local `assets/` directory and credentials outside version control.
