"""Weekly client coverage summary (demo).

Pipeline: load articles -> remove duplicates -> analyse each article with an LLM
(structured output) -> roll up per client -> write a markdown report.

Run without an API key:   python coverage_summary.py --mock
Run with Gemini:          GEMINI_API_KEY=... python coverage_summary.py
Check sentiment accuracy: python coverage_summary.py --mock --eval
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class ArticleAnalysis(BaseModel):
    """Fixed schema the LLM must fill in for every article."""

    sentiment: Literal["positive", "neutral", "negative"]
    theme: str
    summary: str
    evidence_quote: str


# ---------- Step 2: clean ----------
def dedupe(articles: list[dict], threshold: float = 0.85) -> list[dict]:
    """Drop near-duplicate stories (syndicated copies) by title similarity.

    Demo-level logic: a production version would also compare article text
    and publication dates.
    """
    kept: list[dict] = []
    for article in articles:
        title = article["title"].lower()
        if any(SequenceMatcher(None, title, k["title"].lower()).ratio() >= threshold for k in kept):
            continue
        kept.append(article)
    return kept


# ---------- Step 3: analyse ----------
PROMPT = """You are analysing a news article for a PR team about the client "{client}".
Return JSON with:
- sentiment: how the article reflects on the client (positive, neutral or negative)
- theme: a 2-4 word topic label, reused across articles about the same story
- summary: one sentence
- evidence_quote: a short phrase copied from the article that supports the sentiment

Title: {title}
Text: {text}"""

NEGATIVE = ("strike", "delay", "cancel", "furious", "stranded", "fail", "fell", "cost", "criticis")
POSITIVE = ("unveil", "new route", "milestone", "award", "best", "strong", "exceeded")
THEMES = {
    "strike disruption": ("strike", "union"),
    "lisbon route": ("lisbon",),
    "fare sale": ("sale", "fares"),
    "awards": ("award",),
}


def analyse_mock(article: dict, client: str) -> ArticleAnalysis:
    """Keyword stand-in so the pipeline runs without an API key. Not a real model."""
    text = f"{article['title']} {article['text']}".lower()
    neg = sum(w in text for w in NEGATIVE)
    pos = sum(w in text for w in POSITIVE)
    sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
    theme = next((name for name, words in THEMES.items() if any(w in text for w in words)), "other news")
    return ArticleAnalysis(
        sentiment=sentiment,
        theme=theme,
        summary=article["title"],
        evidence_quote=article["text"].split(".")[0],
    )


def analyse_llm(gemini, model: str, article: dict, client: str) -> ArticleAnalysis:
    """Ask Gemini for structured output; retry once if the JSON fails validation."""
    prompt = PROMPT.format(client=client, title=article["title"], text=article["text"])
    last_error = None
    for _ in range(2):
        response = gemini.models.generate_content(
            model=model,
            contents=prompt,
            config={"response_mime_type": "application/json", "response_schema": ArticleAnalysis},
        )
        try:
            return ArticleAnalysis.model_validate_json(response.text)
        except ValidationError as exc:
            last_error = exc
    raise RuntimeError(f"Model returned invalid output twice: {last_error}")


# ---------- Step 4: roll up ----------
def rollup(analyses: list[tuple[dict, ArticleAnalysis]]) -> dict:
    """Counts are computed in code, not by the LLM, so they are always correct."""
    sentiments = Counter(a.sentiment for _, a in analyses)
    themes = Counter(a.theme for _, a in analyses)
    net = sentiments["positive"] - sentiments["negative"]
    overall = "positive" if net > 1 else "negative" if net < -1 else "mixed"

    # "Read first": the most-covered negative story, otherwise the most-covered story.
    neg_themes = Counter(a.theme for _, a in analyses if a.sentiment == "negative")
    focus = (neg_themes or themes).most_common(1)[0][0]
    read_first = next((art for art, a in analyses if a.theme == focus), None)
    return {
        "total": len(analyses),
        "sentiments": dict(sentiments),
        "themes": themes.most_common(),
        "overall": overall,
        "read_first": read_first,
    }


def narrative(gemini, model: str, client: str, summary: dict, mock: bool) -> str:
    facts = (
        f"{summary['total']} articles; sentiment {summary['sentiments']}; "
        f"themes {summary['themes']}; overall {summary['overall']}."
    )
    if mock:
        top = summary["themes"][0][0]
        return f"Coverage of {client} was {summary['overall']} this week, led by '{top}'. ({facts})"
    response = gemini.models.generate_content(
        model=model,
        contents=f"Write a 3-sentence weekly overview for the {client} account team. "
        f"Use only these facts, do not add numbers:\n{facts}",
    )
    return response.text.strip()


# ---------- Step 5: publish ----------
def write_report(client: str, summary: dict, analyses, overview: str, removed: int, path: Path) -> None:
    lines = [
        f"# {client} coverage: week of {date.today():%d %b %Y}",
        "",
        overview,
        "",
        f"**Articles:** {summary['total']} unique ({removed} duplicate(s) removed)  ",
        f"**Sentiment:** {summary['sentiments']}  ",
        f"**Overall:** {summary['overall']}",
        "",
        "## Themes",
        *[f"- {name}: {count}" for name, count in summary["themes"]],
        "",
    ]
    if summary["read_first"]:
        rf = summary["read_first"]
        lines += ["## Read first", f"- [{rf['source']}] {rf['title']}", ""]
    lines += ["## All articles", "", "| Source | Sentiment | Theme | Summary |", "|---|---|---|---|"]
    lines += [f"| {art['source']} | {a.sentiment} | {a.theme} | {a.summary} |" for art, a in analyses]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(analyses) -> None:
    """Compare sentiment against hand-labelled expected values (a mini golden set)."""
    labelled = [(art, a) for art, a in analyses if "expected_sentiment" in art]
    wrong = [(art, a) for art, a in labelled if a.sentiment != art["expected_sentiment"]]
    accuracy = 1 - len(wrong) / len(labelled)
    print(f"Sentiment accuracy vs labels: {accuracy:.0%} ({len(labelled) - len(wrong)}/{len(labelled)})")
    for art, a in wrong:
        print(f"  MISS: '{art['title']}' expected {art['expected_sentiment']}, got {a.sentiment}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/sample_articles.json")
    parser.add_argument("--out", default=f"output/coverage_{date.today():%Y-%m-%d}.md")
    parser.add_argument("--mock", action="store_true", help="use keyword stand-in instead of Gemini")
    parser.add_argument("--eval", action="store_true", help="report accuracy against labelled sentiment")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    client, articles = data["client"], data["articles"]

    unique = dedupe(articles)
    removed = len(articles) - len(unique)

    gemini = None
    if not args.mock:
        if not os.environ.get("GEMINI_API_KEY"):
            print("Set GEMINI_API_KEY, or run with --mock.", file=sys.stderr)
            return 1
        from google import genai  # imported late so --mock needs no extra install

        gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    analyses = [
        (art, analyse_mock(art, client) if args.mock else analyse_llm(gemini, DEFAULT_MODEL, art, client))
        for art in unique
    ]
    summary = rollup(analyses)
    overview = narrative(gemini, DEFAULT_MODEL, client, summary, args.mock)
    write_report(client, summary, analyses, overview, removed, Path(args.out))
    print(f"Wrote {args.out} ({summary['total']} articles, {removed} duplicates removed)")
    if args.eval:
        evaluate(analyses)
    return 0


if __name__ == "__main__":
    sys.exit(main())
