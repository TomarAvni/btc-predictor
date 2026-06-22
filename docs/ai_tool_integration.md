# AI Tool Integration Notes

This project can benefit from AI tooling, but prediction quality must stay gated
by time-aware evidence. New tools should improve evals, workflow, or isolated
research before they influence live forecasts.

## Tool Decisions

| Tool | Upstream | License | Decision |
| --- | --- | --- | --- |
| PromptFoo | `promptfoo/promptfoo` | MIT | Use for grounded tweet-reader evals. |
| Agency Agents | `msitarzewski/agency-agents` | MIT | Use as inspiration for lightweight repo rules only. |
| MiroFish | `666ghj/MiroFish` | AGPL-3.0 | Sandbox only; do not import into runtime. |
| NanoChat | `karpathy/nanochat` | MIT | No production use now; GPU-heavy and not needed for tweet extraction. |
| Impeccable | `pbakaus/impeccable` | Apache-2.0 | Dev-only dashboard/design audit. |
| Heretic | `p-e-w/heretic` | AGPL-3.0 | Do not integrate. It does not fit the project goals. |
| OpenViking | `volcengine/OpenViking` | AGPL-3.0 main project; Apache-2.0 CLI/examples | Sandbox only; evaluate license and service boundaries before use. |

## Anti-Overfitting Rules

- Use chronological splits and walk-forward validation for market data.
- Keep a frozen holdout that is not used for feature selection, prompt tuning,
  model selection, or repeated leaderboard-style iteration.
- Compare candidates against the current serving baseline and a naive baseline.
- Track directional accuracy, Brier/calibration, trading drawdown impact, and
  per-horizon behavior.
- Promote only through report-only, dry-run, and guarded-promotion stages.
- Keep rollback simple: `data/validation/models/` remains the serving source of
  truth until a candidate passes gates.

## Current Integration Points

- `evals/promptfoo/` contains the PromptFoo eval suite for
  `GroundedTweetReader`. If local `npx` package startup stalls, use
  `python -m unittest tests.test_promptfoo_eval_harness` as the deterministic
  smoke check.
- `.cursor/rules/btc-predictor-invariants.mdc` gives future AI agents the core
  invariants to preserve.
- `experiments/` is reserved for sandbox-only work that must not alter live
  predictions without scored evidence.

## Dashboard Audit

Impeccable is suitable as a dev-only detector for `dashboard/` because it is
permissively licensed and does not need to touch prediction logic. Run it
manually when making Streamlit UI changes:

```bash
npx impeccable@latest detect dashboard --fast --json
```

Do not make this a required CI gate until the CLI is stable in the local and
GitHub Actions environments. The first implementation attempt stalled during
`npx` package startup before producing findings, so no dashboard code changes
were made from it.
