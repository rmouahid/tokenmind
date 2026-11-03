# TOKENMIND

> What if LLMs didn't need human language to communicate?

TOKENMIND is an experimental engine that looks for the most token-efficient
way to give information to a large language model without degrading its
answers. It renders the same facts as natural text, condensed text, JSON,
semantic triples, a custom symbolic notation and hybrids, counts their tokens
with several tokenizers, asks several models to answer questions about them,
and selects the cheapest representation that keeps answers correct.

## Hypotheses

- **H1** — the optimal representation depends on the tokenizer and the model.
- **H2** — the optimal representation depends on the task.

The full protocol is in [docs/design.md](docs/design.md).

## Status

Work in progress. Results and charts will be published in this repository
once the benchmark has run.

## Development

```bash
poetry install
poetry run pytest -q
poetry run ruff check . && poetry run ruff format --check .
```

## License

MIT
