# TOKENMIND — experimental design

## Question

Can an LLM be given the same information in fewer tokens than natural
language, without losing answer quality? And if so, which representation
wins, for which model, and for which task?

Showing that compression is possible is not the goal. The goal is to map
**which representations work, for which models and tasks, and at what
compression level comprehension breaks down**.

## Hypotheses

- **H1** — the optimal representation depends on the tokenizer and the model.
- **H2** — the optimal representation depends on the task asked of the model.

Both are tested against the null hypothesis that a single representation is
optimal everywhere.

## Pipeline

```
dataset (facts + questions + gold answers)
   └─ renderers ──► representations × compression levels
        └─ tokenizers ──► token counts (per tokenizer)
        └─ models ──► answers ──► grader ──► accuracy
   └─ optimizer ──► cheapest representation with accuracy ≥ fidelity target
   └─ report ──► tables + charts (H1, H2, degradation curves)
```

### 1. Dataset: synthetic, seeded, exactly gradable

Each item is a small *world*: a set of structured facts (entities with
attributes and relations, e.g. people, companies, cities, products) drawn
from a seeded generator. Synthetic worlds avoid two traps: the model cannot
answer from memorised knowledge (contamination), and every answer has an
exact gold value.

Task families (H2):

| Task | Example question | Answer |
| --- | --- | --- |
| Lookup | What is Lena Moreau's role? | single value |
| Multi-hop | In which city is the employer of Lena Moreau? | single value |
| Aggregation | How many employees of Kestrel earn above 50k? | number |
| Comparison | Who is older, Lena or Tom? | entity |
| Filtering | List the products launched after 2021. | set |

### 2. Representations

The same facts are rendered as:

1. **Natural** — fluent prose (baseline).
2. **Condensed** — telegraphic prose, no function words.
3. **JSON** — minified and indented variants.
4. **Triples** — `subject | predicate | object` lines.
5. **Symbolic** — a custom compact notation (short keys, separators,
   abbreviations).
6. **Hybrid** — e.g. table header once, then rows; or symbolic with a
   natural-language legend.

Variants come from deterministic templates first; an LLM rewriter produces
extra candidates (for instance "compress this as much as possible while
keeping every fact"), which are kept only if a fact-coverage check passes.

**Fair accounting:** any legend, schema or decoding instruction a
representation needs is part of its prompt and counted in its tokens.

### 3. Compression levels

Each representation family is swept over increasing compression levels
(dropping whitespace, shortening keys, abbreviating values, removing
redundancy) to draw a *tokens vs accuracy* curve and locate the point where
accuracy falls off.

### 4. Tokenizers (H1)

Token counts are measured with several tokenizers: OpenAI `o200k_base` and
`cl100k_base` (tiktoken) and open-model tokenizers (Llama 3, Qwen 2.5,
Mistral, Gemma) loaded from Hugging Face. Only tokenizer files are
downloaded, no weights.

### 5. Models

Any OpenAI-compatible chat endpoint (OpenRouter, a local Ollama or
llama.cpp server, OpenAI). Target set: at least 4 models from different
families and sizes. Temperature 0. Every response is cached on disk, keyed
by model + prompt, so a benchmark can be re-run and re-graded for free.

### 6. Grading

Exact match after normalisation (case, whitespace, number formats); set
answers by set equality; no LLM judge, so grading is deterministic.

### 7. Optimisation

For a (model, task) pair, the selected representation is the one with the
fewest prompt tokens whose accuracy is at least `fidelity × accuracy(natural)`
(default fidelity 0.98). The report also draws the Pareto frontier of tokens
vs accuracy.

## Credibility checklist

- Fixed seeds; dataset, prompts and cached responses are committed or
  reproducible from one command.
- Enough items per cell (≥ 100) to report 95 % confidence intervals.
- Token savings reported per tokenizer, including legend overhead.
- Negative results reported as they are.

## Constraints

- Python 3.12, around 500–700 lines of library code.
- Runs on a laptop: no model weights needed, only API calls or a local
  server.
- Open source (MIT), benchmark results and charts in the repository.

## Out of scope

- Fine-tuning a model on a new language.
- Compressing the model's *output*: only the input representation is studied.
