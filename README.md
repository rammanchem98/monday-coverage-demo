# Monday Coverage Summary (demo)

A small, runnable demo of an automated weekly client-coverage summary: pull a week of
news articles about a client, remove duplicates, have an LLM judge each one, and write a
one-page summary the account team can open on Monday morning.

All data here is **fictional sample data** (an invented airline, "SkyJet").

## How it works

```
collect -> clean (dedupe) -> analyse (LLM, fixed schema) -> roll up -> publish
```

| Step | What it does |
|---|---|
| Collect | Reads `data/sample_articles.json`. In production this would be a media-monitoring feed or news API. |
| Clean | Drops syndicated copies using title similarity. |
| Analyse | Gemini returns sentiment, theme, one-line summary and an evidence quote per article, validated against a Pydantic schema (one retry on invalid output). |
| Roll up | Counts and the "read first" story are computed in code, so the numbers are always exact. The LLM only writes the overview sentences from those facts. |
| Publish | Writes a markdown report to `output/`. In production this would go to a shared doc through its API. |

## Run it

```bash
pip install -r requirements.txt

python coverage_summary.py --mock --eval        # no API key needed
GEMINI_API_KEY=... python coverage_summary.py --eval   # real model
pytest                                          # unit tests
```

`--mock` uses a keyword stand-in, not a real model, so you can see the pipeline and output
without a key. `--eval` compares sentiment to hand-labelled expectations in the sample
file, a mini golden set. The mock scores 85% (it misreads two neutral explainer articles
as negative because they mention the strike), which is the kind of error the check is there
to catch.

See `output/example_report.md` for a sample report.

## Running it every Monday

Schedule `coverage_summary.py` with cron, GitHub Actions (`on: schedule`), or a cloud
scheduler triggering a serverless job. Keep the API key in a secrets manager, alert if
the job fails or the feed is empty, and have a person spot-check low-confidence items.

## Limits of this demo

- Sample data and a stub feed, not a live media-monitoring source.
- Dedupe compares titles only; production should also compare text and dates.
- Publishing is a markdown file, not an API write to a shared doc.
- Sentiment accuracy for the real model depends on your labelled set; run `--eval` on it.
