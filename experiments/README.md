# Experiments

This directory is for sandbox-only research tools. Nothing here should affect
live predictions, serving artifacts, or trading behavior without passing the
same held-out scoring and promotion gates as production model changes.

## MiroFish / OpenViking Boundary

MiroFish and the main OpenViking project are AGPL-licensed. Do not import their
packages or vendor their source into this repository. Keep experiments as data
exports or process-isolated local runs unless licensing is reviewed again.

## Sentiment Sandbox Export

Generate seed material from this repo's own sentiment memory:

```bash
python experiments/export_sentiment_sandbox.py
```

The output lands in `experiments/output/`, which is ignored by git. It is safe
to pass that JSON file to an external simulator or context-memory prototype for
manual research. Results remain qualitative until validated against mature
prediction scores.
