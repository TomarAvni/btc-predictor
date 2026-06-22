# PromptFoo Evals

This suite checks the grounded tweet reader contract used by
`src/features/tweet_llm_reader.py`.

## What It Guards

- Output is valid JSON with the expected extraction fields.
- Every extraction cites tweet ids from the current batch.
- The reader stays descriptive and does not forecast price.
- Sentiment sign, self-reported intent, event flags, and relevance stay stable on representative examples.
- Cases are split into `dev` and `frozen`; tune prompts on `dev` only.

## Run Locally

Default mode is deterministic and does not spend API credits:

```bash
npx promptfoo@latest eval -c evals/promptfoo/promptfooconfig.yaml
```

If local `npx` package startup stalls, validate the same provider/assertion
contract without Node:

```bash
python -m unittest tests.test_promptfoo_eval_harness
```

For live LLM evaluation, temporarily set `allow_live: true` in
`promptfooconfig.yaml` and provide `LLM_API_KEY`. Do not tune repeatedly against
the frozen cases; they are the overfitting guard.

## Promotion Rule

Prompt/schema changes should pass the existing Python sentiment tests and this
PromptFoo suite before they are considered for CI gating or production use.
