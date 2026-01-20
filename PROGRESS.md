# PROGRESS

## Milestones
- Initial end-to-end EmbeddingGemma LoRA workflow and notebooks
- Core embedding pipeline in `src/models/embedding_pipeline.py`
- Unit tests for model pipeline and utility components

## Current Focus
- Improve auth ergonomics for gated HF models
- Expand unit test coverage for pipeline behavior

## Decisions
- Use `HF_TOKEN` from the environment and pass it into Hugging Face
  `from_pretrained` calls to avoid hardcoding secrets in source.
- Add `01a_Environment_Variables.ipynb` to validate HF auth and downloads
  without using the Hugging Face CLI.

## Backlog
- Replace hardcoded model IDs with typed config + YAML
- Add token accounting + timing logs under `outputs/logs/`
- Flesh out placeholder tests in `tests/test_embedding_pipeline.py`
