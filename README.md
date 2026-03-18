# AI-Taste-Training

This repository contains a reusable training and validation framework for research-article classification and prompt-based evaluation. The framework supports both API-based fine-tuning and local DeepSpeed training, with a shared data transformation layer and validation pipeline.

## Scope

The open-source-ready part of the codebase is primarily:

- `src/core`
- `src/practices/article`
- `scripts/sft`
- `scripts/validation`
- `test`

The framework is built around four layers:

1. `src/core`
   Provides generic infrastructure for config loading, dataloaders, inference, validation metrics, and SFT backends.
2. `src/practices/article`
   Implements the article-specific prompt registry, data transformation, filtering, validation CLI, SFT CLI, and document models.
3. `scripts`
   Contains runnable shell entrypoints for validation and training workflows.
4. `test`
   Contains integration-style tests and utility probes for inference, training, data handling, and provider-specific behavior.

## Main Entry Points

### Validation

Run validation through:

```bash
python -m src.practices.article.validation \
  --model <model_name> \
  --provider <provider_name> \
  --prompt <prompt_name> \
  --entry rq_with_context \
  --data_source mongodb \
  --db_name <db_name> \
  --split validate \
  --subjects ECONOMICS SOCIOLOGY
```

Useful shell wrappers:

- `scripts/validation/val.sh`
- `scripts/validation/val_lp.sh`
- `scripts/validation/val_ckpt.sh`

### SFT

Run SFT through:

```bash
python -m src.practices.article.sft \
  --trainer deepspeed \
  --data_source mongodb \
  --db_name <db_name> \
  --split train \
  --subjects ECONOMICS SOCIOLOGY \
  --prompt social_science_rqcontext \
  --entry rq_with_context \
  --target_field rank \
  --model_path /path/to/base-model \
  --output_dir ./finetune/run_name
```

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

This layout is intentional:

- `.gitignore` currently ignores `/assets`
- `.gitignore` also ignores `*.toml`
- therefore the public templates must not live inside `assets/` and must not end with plain `.toml`

Recommended local setup for users:

1. Copy `assets_example/model.toml.example` to `assets/model.toml`
2. Copy `assets_example/database.toml.example` to `assets/database.toml`
3. Fill in local credentials and private paths only in those ignored local files

Environment variables are still supported by some Python utilities and tests, but the shell training/validation entrypoints now read database settings from `assets/database.toml`.

## Tests

The `test` directory is mixed:

- some files are reusable framework tests
- some are local probes or one-off provider checks
- some currently assume private credentials or private infrastructure

Before publishing, keep only tests that:

- do not require private databases
- do not embed real API keys
- do not depend on local absolute paths

## Release Checklist

Before open-sourcing this framework:

1. Remove real API keys from `assets/model.toml` and `assets/model_old.toml`.
2. Remove real database credentials from `assets/database.toml`.
3. Replace hard-coded MongoDB connection strings in shell scripts with environment-variable-based wiring.
4. Remove hard-coded secrets from tests and temporary scripts.
5. Remove local absolute paths such as `/workspace/...` and `C:\\Users\\...` where they are not required.
6. Keep only public-safe examples in `scripts` and `test`.

## Recommended Minimal Public Layout

If publishing a clean version, keep:

- `src/core`
- `src/practices/article`
- selected `scripts/sft`
- selected `scripts/validation`
- sanitized `test`
- `assets_example/*.toml.example`

Archive or remove:

- private data-processing scripts
- internal migration scripts
- scripts with hard-coded production DB settings
- tests containing real secrets or private hosts

## Known Security Risks In Current Repo

Current repository state is not safe to publish as-is if private local config files are copied into the release package. The tracked code is now structured so that public examples can live outside `assets/`, but you should still verify that no real local config files are bundled.

- real API keys in `assets/model.toml`
- real API keys in `assets/model_old.toml`
- real MongoDB credentials in `assets/database.toml`
- hard-coded MongoDB connection strings in multiple shell scripts
- hard-coded credentials in some test files

Sanitize those first, then publish.
